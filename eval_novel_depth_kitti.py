import time, argparse, os.path as osp
import os, math, pickle
import torch, numpy as np
import torch.distributed as dist
import torch.nn.functional as F

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

from utils.metric_util import compute_depth_errors
from utils.config_tools import modify_for_eval


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, 'kitti', True)
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
        cfg.dump(osp.join(args.work_dir, 'eval_novel_depth_' + osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_novel_depth_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    # logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(args)

    import model
    from dataset import get_dataloader

    # build model
    cfg.model.head.return_max_depth = True
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed,)

    # get optimizer, loss, scheduler
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
        print(f'successfully resumed from epoch {ckpt["epoch"]}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()

    with torch.no_grad():
        for i_iter_val, (input_imgs, anchor_imgs, img_metas) in enumerate(val_dataset_loader):
            
            input_imgs = input_imgs.cuda()
            anchor_imgs = anchor_imgs.cuda()
            
            with torch.cuda.amp.autocast(amp):

                result_dict = my_model(imgs=input_imgs, metas=img_metas, prepare=True)

            frame_id = img_metas[0]['token']
            sequence = img_metas[0]['sequence']

            save_dir = os.path.join(args.work_dir, "depth_metrics", sequence)
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = os.path.join(save_dir,"{}.npy".format(frame_id))
            if os.path.exists(save_filepath):
                continue
            
            source_distances = img_metas[0]["frame_dists"]
            loc2d_with_depths = img_metas[0]['depth_loc'] # [N, n, 2] * S
            lidar_depths = img_metas[0]['depth_gt']       # [N, n] * S
            lidar_depths_mask = img_metas[0]['depth_mask'] # [N, n] * S

            agg_depth_errors = {}
            n_frames = {}

            for source_id in range(len(source_distances)):
                
                source_distance = source_distances[source_id]
                loc2d_with_depth = input_imgs.new_tensor(loc2d_with_depths[source_id])
                lidar_depth = input_imgs.new_tensor(lidar_depths[source_id])
                lidar_depth_mask = torch.from_numpy(lidar_depths_mask[source_id]).cuda()

                gt_depth_infer = lidar_depth[0][lidar_depth_mask[0]] # n

                img_metas[0].update({
                    'render_img2lidar': img_metas[0]['temImg2lidars'][source_id]})
                
                render_out_dict = my_model.module.head.render(metas=img_metas, batch=args.batch)

                if args.depth_tgt == 'raw':
                    depth_key = 'ms_depths'
                elif args.depth_tgt == 'max':
                    depth_key = 'ms_max_depths'
                
                pred_depth_infer = render_out_dict[depth_key][0] # B, N, R
                pred_depth_infer = pred_depth_infer.reshape(1, 1, *cfg.num_rays)
                pred_depth_infer = F.grid_sample(
                    pred_depth_infer, # 1, 1, H, W
                    loc2d_with_depth[None, ...] * 2 - 1, # 1, 1, n, 2
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True).squeeze() # n
                assert len(pred_depth_infer.shape) == 1
                pred_depth_infer = pred_depth_infer[lidar_depth_mask[0]]
                assert len(pred_depth_infer) == len(gt_depth_infer)
                
                depth_errors = evaluate_depth(gt_depth_infer, pred_depth_infer)
                
                k = math.ceil(source_distance)
                
                if k not in agg_depth_errors:
                    agg_depth_errors[k] = depth_errors
                    n_frames[k] = 1
                else: 
                    agg_depth_errors[k] += depth_errors
                    n_frames[k] += 1

            out_dict = {
                "depth_errors": agg_depth_errors,
                "n_frames": n_frames
            }
            with open(save_filepath, "wb") as output_file:
                pickle.dump(out_dict, output_file)
                logger.info("Saved to" + save_filepath)

            
            logger.info("=================")
            logger.info("==== Frame {} ====".format(frame_id))
            logger.info("=================")
            print_metrics(agg_depth_errors, n_frames, logger)
    
    if not distributed or dist.get_rank() == 0:
        from tqdm import tqdm
    
        train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=False)

        cnt = 0
        agg_depth_errors = {}
        agg_n_frames = {}
        for i_iter_val, (input_imgs, anchor_imgs, img_metas) in enumerate(tqdm(val_dataset_loader)):
            cnt += 1
                    
            frame_id = img_metas[0]['token']
            sequence = img_metas[0]['sequence']
            
            save_dir = os.path.join(args.work_dir, "depth_metrics", sequence)
            save_filepath = os.path.join(save_dir, "{}.npy".format(frame_id))

            with open(save_filepath, "rb") as handle:
                data = pickle.load(handle)
            depth_errors = data["depth_errors"]
            n_frames = data["n_frames"]

            for k in depth_errors:  
                if k not in agg_depth_errors:
                    agg_depth_errors[k] = depth_errors[k]
                    agg_n_frames[k] = n_frames[k]
                else: 
                    agg_depth_errors[k] += depth_errors[k]
                    agg_n_frames[k] += n_frames[k]
            
            if cnt % 20 == 0:
                logger.info("=================")
                logger.info("==== batch {} ====".format(cnt))
                logger.info("=================")
                print_metrics(agg_depth_errors, agg_n_frames, logger)
        logger.info("=================")
        logger.info("====== TotalS ======")
        logger.info("=================")
        print_metrics(agg_depth_errors, agg_n_frames, logger)
    

def evaluate_depth(gt_depth, pred_depth):
    depth_errors = []

    
    depth_error = compute_depth_errors(
        gt=gt_depth.reshape(-1).detach().cpu().numpy(),
        pred=pred_depth.reshape(-1).detach().cpu().numpy(),
    )
    # print(depth_error)
    depth_errors.append(depth_error)

    
    agg_depth_errors = np.array(depth_errors).sum(0)

    return agg_depth_errors


def print_metrics(agg_depth_errors, n_frames, logger):
    logger.info("|distance|abs_rel |sq_rel  |rmse     |rmse_log|a1      |a2      |a3      |n_frames|")

    total_depth_errors = None
    total_frame = 0 
    for distance in sorted(agg_depth_errors):
        if total_depth_errors is None:
            total_depth_errors = np.copy(agg_depth_errors[distance])
        else:
            total_depth_errors = total_depth_errors + agg_depth_errors[distance]
        metric_list = ["abs_rel", "sq_rel",
                        "rmse", "rmse_log", "a1", "a2", "a3"]
        logger.info("|{:08d}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:08d}|".format(
            distance,
            agg_depth_errors[distance][0]/n_frames[distance],
            agg_depth_errors[distance][1]/n_frames[distance],
            agg_depth_errors[distance][2]/n_frames[distance],
            agg_depth_errors[distance][3]/n_frames[distance],
            agg_depth_errors[distance][4]/n_frames[distance],
            agg_depth_errors[distance][5]/n_frames[distance],
            agg_depth_errors[distance][6]/n_frames[distance],
            n_frames[distance]
        ))
        total_frame += n_frames[distance]
    logger.info("|{}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:08d}|".format(
            "All     ",
            total_depth_errors[0]/total_frame,
            total_depth_errors[1]/total_frame,
            total_depth_errors[2]/total_frame,
            total_depth_errors[3]/total_frame,
            total_depth_errors[4]/total_frame,
            total_depth_errors[5]/total_frame,
            total_depth_errors[6]/total_frame,
            total_frame
        ))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--depth-tgt', type=str, default='raw')
    parser.add_argument('--batch', type=int, default=0)
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
