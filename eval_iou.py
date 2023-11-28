import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')

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

from nuscenes import NuScenes

def pass_print(*args, **kwargs):
    pass

def read_occ3d_label(metas, nusc):
    token = metas['token']
    scene_token = nusc.get('sample', token)['scene_token']
    scene_name = nusc.get('scene', scene_token)['name']
    label_file = f'data/occ3d/gts/{scene_name}/{token}/labels.npz'
    label = np.load(label_file)
    return label

def read_openoccupancy_label(metas, nusc):
    # import pdb; pdb.set_trace()
    token = metas['token']
    scene_token = nusc.get('sample', token)['scene_token']
    lidar_token = nusc.get('sample', token)['data']['LIDAR_TOP']
    label_file = f'data/nuScenes-Occupancy/scene_{scene_token}/occupancy/{lidar_token}.npy'
    label = np.load(label_file)
    return torch.from_numpy(label[:, :3])

def main(local_rank, args):
    try:
        nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
    except:
        nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', load_lidar_task=False)
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, 'nuscenes')
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
        cfg.dump(osp.join(args.work_dir, 'eval_iou_' + osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_iou_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger

    # build model
    import model
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
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
    from utils.metric_util import IoU, MeanIoU, openseed2nuscenes
    print_freq = cfg.print_freq            
    my_model.eval()
    iou_metric = MeanIoU([1], 0, ['occupied'], True, 0)
    iou_metric.reset()
    if args.sem:
        miou_metric = MeanIoU(
            list(range(1, 17)),
            0,
            ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
            True, 0)
        miou_metric.reset()

    if args.occ3d:
        xx = torch.linspace(-40.0, 40.0, 200)
        yy = torch.linspace(-40.0, 40.0, 200)
        zz = torch.linspace(-1.0, 5.4, 16)
        xyz = torch.stack([
            xx[:, None, None].expand(-1, 200, 16),
            yy[None, :, None].expand(200, -1, 16),
            zz[None, None, :].expand(200, 200, -1),
            # xx[None, :, None].expand(200, -1, 16),
            # yy[:, None, None].expand(-1, 200, 16),
            # zz[None, None, :].expand(200, 200, -1),
            torch.ones(200, 200, 16)
        ], dim=-1) # 200, 200, 16, 4
        xyz = xyz.cuda()

    with torch.no_grad():
        for i_iter_val, (input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, \
                        img_metas, curr_aug, prev_aug, next_aug) in enumerate(val_dataset_loader):
            
            input_imgs = input_imgs.cuda()
            
            with torch.cuda.amp.autocast(amp):
                
                if not args.occ3d:
                    point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
                elif args.scene_size == 0:
                    point_cloud_range = [-51.2, -51.2, -4, 51.2, 51.2, 4]
                    expansion = [102.4, 102.4, 8]
                elif args.scene_size == 1:
                    point_cloud_range = [-40.0, -40.0, -2.8, 40.0, 40.0, 3.6]
                    expansion = [80.0, 80.0, 6.4]
                elif args.scene_size == 2:
                    point_cloud_range = [-40.0, -40.0, -3.1, 40.0, 40.0, 3.9]
                    expansion = [80.0, 80.0, 7.0]
                elif args.scene_size == 3:
                    point_cloud_range = [-40.0, -40.0, -3.2, 40.0, 40.0, 4.0]
                    expansion = [80.0, 80.0, 7.2]
                elif args.scene_size == 4:
                    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
                    expansion = [80.0, 80.0, 6.4]
                elif args.scene_size == 5:
                    point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
                    expansion = [102.4, 102.4, 8]
                elif args.scene_size == 6:
                    point_cloud_range = [-51.2, -51.2, -4, 51.2, 51.2, 5]
                    expansion = [102.4, 102.4, 9]

                result_dict = my_model(
                    imgs=input_imgs, 
                    metas=img_metas,
                    aabb=point_cloud_range,
                    resolution=args.resolution,
                    occ_only=True)
                if args.density:
                    result_dict['sdf'] = result_dict['sigma']
                    pred_occ = (result_dict['sdf'] >= args.thresh).to(torch.int)
                else:
                    pred_occ = (result_dict['sdf'] <= args.thresh).to(torch.int)
                
                if args.occ3d:
                    ego2lidar = img_metas[0]['ego2lidar']
                    ego2lidar = xyz.new_tensor(ego2lidar) # 4, 4
                    lidar_points = torch.matmul(ego2lidar.unsqueeze(0), xyz.reshape(-1, 4, 1))
                    lidar_points = lidar_points.squeeze(-1)[:, :3] # 200*200*16, 3
                    lidar_points[:, 0] = (lidar_points[:, 0] - point_cloud_range[0]) / expansion[0]
                    lidar_points[:, 1] = (lidar_points[:, 1] - point_cloud_range[1]) / expansion[1]
                    lidar_points[:, 2] = (lidar_points[:, 2] - point_cloud_range[2]) / expansion[2]
                    lidar_points = lidar_points.reshape(1, 200, 200, 16, 3)
                    sampled_sdf = F.grid_sample(
                        result_dict['sdf'][None, None, ...], # 1, 1, H, W, D
                        lidar_points[..., [2, 0, 1]] * 2 - 1,
                        mode='bilinear',
                        align_corners=True) # 1, 1, 200, 200, 16
                    if args.density:
                        pred_occ = (sampled_sdf.squeeze() >= args.thresh).to(torch.int)
                    else:
                        pred_occ = (sampled_sdf.squeeze() <= args.thresh).to(torch.int)
                    pred_occ[..., 12:] = 0
                    pred_occ[:6, ...] = 0
                    pred_occ[-6:, ...] = 0
                    pred_occ[:, :6, :] = 0
                    pred_occ[:, -6:, :] = 0
                    pred_occ_iou = pred_occ
                    
                    gt_meta = read_occ3d_label(img_metas[0], nusc)
                    masks = torch.from_numpy(gt_meta['mask_camera']).bool().cuda()
                    gt_occ_raw = torch.from_numpy(gt_meta['semantics']).cuda()
                    gt_occ_raw[gt_occ_raw == 17] = 0
                    gt_occ_iou = (gt_occ_raw > 0).to(torch.int)
                    if args.sem:
                        gt_occ_miou = gt_occ_raw
                        sem = result_dict['logits'].permute(3, 0, 1, 2) # C, H, W, D
                        sampled_sem = F.grid_sample(
                            sem[None, ...], # 1, C, H, W, D
                            lidar_points[..., [2, 0, 1]] * 2 - 1,
                            mode='bilinear',
                            align_corners=True) # 1, C, 200, 200, 16
                        sampled_sem = torch.argmax(sampled_sem, dim=1).squeeze()
                        sem = openseed2nuscenes(sampled_sem)
                        pred_occ_miou = pred_occ * sem
                else:                    
                    pred_occ[..., -4:] = 0
                    pred_occ[..., :5] = 0
                    pred_occ[:6, ...] = 0
                    pred_occ[-6:, ...] = 0
                    pred_occ[:, :6, :] = 0
                    pred_occ[:, -6:, :] = 0
                    pred_occ_iou = pred_occ

                    gt_occ = read_openoccupancy_label(img_metas[0], nusc).cuda()
                    gt_occ = gt_occ[:, [1, 2, 0]]
                    gt_occ_iou = torch.zeros(512, 512, 40, device=pred_occ.device, dtype=torch.int)
                    gt_occ_iou[gt_occ.transpose(0, 1).tolist()] = 1
                    masks = None

                    if args.sem:
                        gt_occ_miou = gt_occ
                        sem = result_dict['sem']
                        sem = openseed2nuscenes(sem)
                        pred_occ_miou = pred_occ * sem
            
            if args.save_sem:
                token = img_metas[0]['token']
                scene_token = nusc.get('sample', token)['scene_token']
                scene_name = nusc.get('scene', scene_token)['name']

                tosave = pred_occ_miou.cpu().numpy().astype(np.uint8)
                save_path = '{}/{}/'.format(scene_name, token)
                save_path = osp.join(args.save_sem, save_path)
                os.makedirs(save_path, exist_ok=True)
                np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=tosave)

            iou_metric._after_step(pred_occ_iou, gt_occ_iou, masks if args.use_mask else None)
            if args.sem:
                miou_metric._after_step(pred_occ_miou, gt_occ_miou, masks if args.use_mask else None)

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d / %5d'%(i_iter_val, len(val_dataset_loader)))

        iou_miou, iou_iou = iou_metric._after_epoch()
        logger.info(f"iou_miou: {iou_miou}, iou_iou: {iou_iou}")
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
    parser.add_argument('--occ3d', action='store_true', default=False)
    parser.add_argument('--resolution', type=float, default=0.2)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--sem', action='store_true', default=False)
    parser.add_argument('--use-mask', action='store_true', default=False)
    parser.add_argument('--scene-size', type=int, default=0)
    parser.add_argument('--density', action='store_true', default=False)
    parser.add_argument('--save-sem', type=str, default='')
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
