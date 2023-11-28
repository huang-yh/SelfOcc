import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

# import model
from dataset import get_dataloader
from utils.config_tools import modify_for_eval
import dataset.kitti.io_data as SemanticKittiIO
KITTI_ROOT = 'data/kitti'


def pass_print(*args, **kwargs):
    pass

def read_semantic_kitti(metas):
    label_path = os.path.join(
        KITTI_ROOT, "dataset/sequences", metas['sequence'], "voxels", "{}.label".format(metas['token']))
    invalid_path = os.path.join(
        KITTI_ROOT, "dataset/sequences", metas['sequence'], "voxels", "{}.invalid".format(metas['token']))

    remap_lut = SemanticKittiIO.get_remap_lut("dataset/kitti/semantic-kitti.yaml")
    LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
    LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
        np.float32
    )  # Remap 20 classes semanticKITTI SSC
    
    LABEL[
        np.isclose(INVALID, 1)
    ] = 255  # Setting to unknown all voxels marked on invalid mask...
    
    LABEL = LABEL.reshape(256, 256, 32)
    return LABEL

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, 'kitti')
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, 'eval_iou_kitti' + osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_iou_kitti_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from utils.metric_util import cityscapes2semantickitti
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    logger.info('done build model')


    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed)

    amp = cfg.get('amp', False)    

    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        print(f'successfully resumed from {cfg.resume_from}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # training
    from utils.metric_util import IoU, MeanIoU
    from utils.scenerf_metric import SSCMetrics
    print_freq = cfg.print_freq            
    my_model.eval()
    iou_metric = IoU()
    iou_metric.reset()
    scenerf_metric = SSCMetrics(2)
    if args.sem:
        miou_metric = MeanIoU(
            list(range(1, 20)),
            0,
            ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
            "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", 
            "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"],
            True, 0)
        miou_metric.reset()
    max_ds, min_ds = [], []

    with torch.no_grad():
        for i_iter_val, (input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, \
                        img_metas, curr_aug, prev_aug, next_aug) in enumerate(val_dataset_loader):
            
            input_imgs = input_imgs.cuda()
            
            with torch.cuda.amp.autocast(amp):

                result_dict = my_model(
                    imgs=input_imgs, 
                    metas=img_metas,
                    aabb=[-25.6, 0, -2.0, 25.6, 51.2, 4.4],
                    resolution=args.resolution,
                    occ_only=True)
                
                pred_occ = (result_dict['sdf'] <= args.thresh).to(torch.int)
                
                # gt_occ_raw for scenerf style iou calculation
                # gt_occ for my own evaluation
                gt_occ_raw = torch.from_numpy(read_semantic_kitti(img_metas[0])).cuda()
                gt_occ_raw = torch.flip(gt_occ_raw, [1])
                gt_occ = gt_occ_raw.clone() # gt_occ = np.copy(gt_occ_raw)
                gt_occ[gt_occ == 255] = 0
                gt_occ = torch.nonzero(gt_occ)

                ## post process
                max_d = gt_occ[:, 2].max()
                min_d = gt_occ[:, 2].min()
                # pred_occ[..., (max_d + 1):] = 0
                # pred_occ[..., :min_d] = 0
                pred_occ[..., 28:] = 0
                # pred_occ[0, ...] = 0
                pred_occ[-6:, ...] = 0
                pred_occ[:, :6, :] = 0
                pred_occ[:, -6:, :] = 0

            iou_metric._after_step(pred_occ, gt_occ)
            gt_occ_scenerf = gt_occ_raw.clone()
            scenerf_metric.add_batch(pred_occ, gt_occ_scenerf)
            if args.sem:
                sem = result_dict['sem']
                sem = cityscapes2semantickitti(sem)
                # gt_occ_raw[gt_occ_raw == 255] = 0
                pred_miou = pred_occ * sem
                miou_metric._after_step(pred_miou, gt_occ_raw, gt_occ_raw != 255)

            max_ds.append(max_d.item())
            min_ds.append(min_d.item())

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d / %5d, max_d %d, min_d %d'%(
                    i_iter_val, len(val_dataset_loader), max_d, min_d))

        iou = iou_metric._after_epoch()
        stats = scenerf_metric.get_stats()
        if not distributed or dist.get_rank() == 0:
            logger.info(f'IoU: {iou}')
            logger.info(f'mean of max_d: {np.mean(max_ds)}')
            logger.info(f'mean of min_d: {np.mean(min_ds)}')

            logger.info("========================")
            logger.info("=========Summary========")   
            logger.info("========================")
            logger.info("==== Whole Scene ====")
            logger.info(f"iou: {stats['iou']}, precision: {stats['precision']}, recall: {stats['recall']}")
        if args.sem:
            miou_miou, miou_iou = miou_metric._after_epoch()
            logger.info(f"miou_miou: {miou_miou}, miou_iou: {miou_iou}")
            

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=float, default=0.2)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--sem', action='store_true', default=False)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.hfai:
        os.environ['HFAI'] = 'true'

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
