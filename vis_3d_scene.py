import os
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=(2560, 1440))
    display.start()
    offscreen = True

from mayavi import mlab
import mayavi
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import os, time, argparse, math, os.path as osp, numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')
import shutil

import mmcv
from mmengine import Config
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

import model
from dataset import get_dataloader
from dataset.dataset_wrapper_temporal import custom_collate_fn_temporal
from utils.config_tools import modify_for_eval
from utils.metric_util import cityscapes2semantickitti, openseed2nuscenes
from vis_pics import create_label_colormap
from PIL import Image

def pass_print(*args, **kwargs):
    pass

KITTI_ROOT = 'data/kitti'
import dataset.kitti.io_data as SemanticKittiIO
def read_semantic_kitti(metas, idx):
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
    LABEL = np.flip(LABEL, 1)

    save_info = f"{metas['sequence']}_{metas['token']}_{idx}.png"
    return LABEL, save_info

def read_occ3d_label(metas, nusc, idx):
    token = metas['token']
    scene_token = nusc.get('sample', token)['scene_token']
    scene_name = nusc.get('scene', scene_token)['name']
    label_file = f'data/occ3d/gts/{scene_name}/{token}/labels.npz'
    label = np.load(label_file)
    save_info = f"{scene_name}_{token}_{idx}.png"
    return label, save_info


def draw(
    voxels,          # semantic occupancy predictions
    grid_coords,
    voxel_size=0.2,  # voxel size in the real world
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
    dataset='nuscenes',
):
    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    
        # draw a simple car at the middle
        # car_vox_range = np.array([
        #     [w//2 - 2 - 4, w//2 - 2 + 4],
        #     [h//2 - 2 - 4, h//2 - 2 + 4],
        #     [z//2 - 2 - 3, z//2 - 2 + 3]
        # ], dtype=np.int)
        # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
        # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
        # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
        # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
        # car_label = np.zeros([8, 8, 6], dtype=np.int)
        # car_label[:3, :, :2] = 17
        # car_label[3:6, :, :2] = 18
        # car_label[6:, :, :2] = 19
        # car_label[:3, :, 2:4] = 18
        # car_label[3:6, :, 2:4] = 19
        # car_label[6:, :, 2:4] = 17
        # car_label[:3, :, 4:] = 19
        # car_label[3:6, :, 4:] = 17
        # car_label[6:, :, 4:] = 18
        # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
        # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
        # grid_coords[car_indexes, 3] = car_label.flatten()
    
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 1],
            fov_voxels[:, 0],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 1],
            fov_voxels[:, 0],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=19 if dataset == 'kitti' else 16, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        if dataset == 'kitti':
            colors = create_label_colormap()
            colors = np.concatenate([colors, np.ones_like(colors[:, :1]) * 255], axis=-1)
        else:
            colors = np.array(
                [
                    [255, 120,  50, 255],       # barrier              orange
                    [255, 192, 203, 255],       # bicycle              pink
                    [255, 255,   0, 255],       # bus                  yellow
                    [  0, 150, 245, 255],       # car                  blue
                    [  0, 255, 255, 255],       # construction_vehicle cyan
                    [255, 127,   0, 255],       # motorcycle           dark orange
                    [255,   0,   0, 255],       # pedestrian           red
                    [255, 240, 150, 255],       # traffic_cone         light yellow
                    [135,  60,   0, 255],       # trailer              brown
                    [160,  32, 240, 255],       # truck                purple                
                    [255,   0, 255, 255],       # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [ 75,   0,  75, 255],       # sidewalk             dard purple
                    [150, 240,  80, 255],       # terrain              light green          
                    [230, 230, 250, 255],       # manmade              white
                    [  0, 175,   0, 255],       # vegetation           green
                    # [  0, 255, 127, 255],       # ego car              dark cyan
                    # [255,  99,  71, 255],       # ego car
                    # [  0, 191, 255, 255]        # ego car
                ]
            ).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    for i, save_dir in enumerate(save_dirs):
        if i < 6:
            scene.camera.position = cam_positions[i] - np.array([0.7, 1.3, -1.])
            scene.camera.focal_point = focal_positions[i] - np.array([0.7, 1.3, -1.])
            scene.camera.view_angle = 35 if i != 3 else 60
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
        elif i == 6:
            # scene.camera.position = [-4.69302904, -52.74874688, 19.16181492]
            # scene.camera.focal_point = [-4.52985313, -51.8233303, 18.81979477]
            # scene.camera.view_angle = 40.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [0.01, 300.]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
            scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        else:
            # scene.camera.position = [91.84365261779985, 87.2356528161641, 86.90232146965226]
            # scene.camera.focal_point = [4.607997894287109, -1.9073486328125e-06, -0.33333325386047363]
            # scene.camera.view_angle = 30.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [33.458354318473965, 299.5433372220855]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
            scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0., 1., 0.]
            scene.camera.clipping_range = [0.01, 400.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()


def main(local_rank, args):
    if args.dataset == 'nuscenes':
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
        for scene_meta in nusc.scene:
            if scene_meta['name'] == args.scene_name:
                scene_token = scene_meta['token']
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, args.dataset)
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
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)
        local_rank = dist.get_rank()

        if dist.get_rank() != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        local_rank = 0

    # configure logger
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, 'eval' + osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval-{timestamp}.log')
    logger = MMLogger(name='selfocc', log_file=log_file, log_level='INFO')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model    
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
        print('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.to(args.device)
        raw_model = my_model
    print('done ddp model')

    cfg.train_wrapper_config.phase = 'val'
    if args.scene_name:
        cfg.val_dataset_config.update({
            'type': 'nuScenes_One_Frame_Sweeps_Dist_Vis',
            'imageset': 'data/nuscenes_infos_val_sweeps_lid.pkl',
            'scene_name': args.scene_name,
            'scene_token': scene_token})
        cfg.val_wrapper_config.update({
            'type': 'tpvformer_dataset_nuscenes_vis',
        })
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed,
        iter_resume=False,
        val_only=True)
    if args.vis_train:
        dataset = train_dataset_loader.dataset
    else:
        dataset = val_dataset_loader.dataset
    
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    epoch = 'last'
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if 'epoch' in ckpt:
            epoch = ckpt['epoch']
        print(raw_model.load_state_dict(state_dict, strict=False))
        print(f'successfully resumed from {cfg.resume_from}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))

    xx = torch.linspace(-40.0, 40.0, 200)
    yy = torch.linspace(-40.0, 40.0, 200)
    zz = torch.linspace(-1.0, 5.4, 16)
    xyz = torch.stack([
        xx[:, None, None].expand(-1, 200, 16),
        yy[None, :, None].expand(200, -1, 16),
        zz[None, None, :].expand(200, 200, -1),
        torch.ones(200, 200, 16)
    ], dim=-1) # 200, 200, 16, 4
    xyz = xyz.to(args.device)

    # eval
    my_model.eval()
    vis_save_path = osp.join(args.work_dir, args.dir_name, f'epoch{epoch}')
    if args.scene_name:
        vis_save_path = osp.join(vis_save_path, args.scene_name)
    clip_dirs = []
    for video_clip in range(8):
        clip_dir = os.path.join(vis_save_path, str(video_clip))
        clip_dirs.append(clip_dir)
        os.makedirs(clip_dir, exist_ok=True)

    os.makedirs(vis_save_path, exist_ok=True)

    # args.frame_idx = range(0, len(dataset), 10)
    # args.frame_idx = range(6000, 3000, -10)
    # args.frame_idx = [2400, 2420, 2490, 2530, 2670, 2680, 2780, 2820, 2950, 3210, 3450, 3510, 3560, 3610, 3730, 4150, 4320, 4330, 4450, 4530, 4580, 4640, 4890, 5040, 5120, 5220, 5370, 5550]
    # args.frame_idx = [0,110,190,270,350,360,380,420,440,480,500,550,560,600,650,710,720,750,770,840,880,930,940,990,1090,1130,1200,1210,1300,1560,1600,1850,1930,1980,2000,2050,2040,2090,2120,2400,2490,2550,2670,2820,3200,3510,3560,3690,3800,4100,4150,4320,4530,4580,4890,5220,5370]
    # args.frame_idx = [2000, 1300, 990, 1600, 1980]

    #kitti
    # args.frame_idx = [640, 0, 130, 260, 330, 470, 770, 730, 750, 570, 600, 400, 510, 100]

    if args.scene_name:
        args.frame_idx = list(range(args.start_idx, len(dataset)))

    with torch.no_grad():
        # num_iters = len(val_dataset_loader) if args.scene_name else len(args.frame_idx)
        # if args.scene_name:
        #     loader_iter = iter(val_dataset_loader)
        # for idx in range(num_iters):
        #     if args.scene_name:
        #         input_imgs, img_metas = next(loader_iter)
        #         real_idx = img_metas[0]['timestamp']
        #     else:
        #         real_idx = args.frame_idx[idx]
        #         one_batch = dataset[real_idx]
        #         input_imgs, img_metas = custom_collate_fn_temporal([one_batch])

        for idx in args.frame_idx:
            one_batch = dataset[idx]
            input_imgs, img_metas = custom_collate_fn_temporal([one_batch])
            real_idx = img_metas[0]['timestamp']

            input_imgs = input_imgs.to(args.device)

            if 'kitti' in args.dataset:
                point_cloud_range = [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]
                resolution = 0.2
            else:
                if args.scene_size == 0:
                    point_cloud_range = [-51.2, -51.2, -4, 51.2, 51.2, 4]
                    resolution = 0.4
                elif args.scene_size == 1:
                    point_cloud_range = [-40.0, -40.0, -2.8, 40.0, 40.0, 3.6]
                    resolution = 0.4
                elif args.scene_size == 2:
                    point_cloud_range = [-40.0, -40.0, -3.1, 40.0, 40.0, 3.9]
                    resolution = 0.4
                elif args.scene_size == 3:
                    point_cloud_range = [-40.0, -40.0, -3.2, 40.0, 40.0, 4.0]
                    resolution = 0.4
                elif args.scene_size == 4:
                    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
                    resolution = 0.4

            if args.model_pred:

                result_dict = my_model(
                    imgs=input_imgs, 
                    metas=img_metas, 
                    occ_only=True,
                    aabb=point_cloud_range,
                    resolution=resolution)
                
                if args.transform:
                    ego2lidar = img_metas[0]['ego2lidar']
                    ego2lidar = xyz.new_tensor(ego2lidar) # 4, 4
                    lidar_points = torch.matmul(ego2lidar.unsqueeze(0), xyz.reshape(-1, 4, 1))
                    lidar_points = lidar_points.squeeze(-1)[:, :3] # 200*200*16, 3
                    # ori_lidar_points = lidar_points.clone()

                    lidar_points[:, 0] = (lidar_points[:, 0] - point_cloud_range[0]) / (point_cloud_range[3] - point_cloud_range[0])
                    lidar_points[:, 1] = (lidar_points[:, 1] - point_cloud_range[1]) / (point_cloud_range[4] - point_cloud_range[1])
                    lidar_points[:, 2] = (lidar_points[:, 2] - point_cloud_range[2]) / (point_cloud_range[5] - point_cloud_range[2])
                    lidar_points = lidar_points.reshape(1, 200, 200, 16, 3)
                    sampled_sdf = F.grid_sample(
                        result_dict['sdf'][None, None, ...], # 1, 1, H, W, D
                        lidar_points[..., [2, 0, 1]] * 2 - 1,
                        mode='bilinear',
                        align_corners=True,
                        padding_mode='border') # 1, 1, 200, 200, 16
                    sdf = sampled_sdf.squeeze().permute(1, 0, 2).cpu()
                else:
                    sdf = result_dict['sdf'].cpu()
                grid_size = sdf.shape
                voxel_density = sdf
                predict_vox = (voxel_density <= args.thresh).to(torch.int)
                predict_vox[:6, ...] = 0
                predict_vox[-6:, ...] = 0
                predict_vox[:, :6, :] = 0
                predict_vox[:, -6:, :] = 0

                print((voxel_density > 1.0).sum())
                print((voxel_density > 2.0).sum())
                print((voxel_density > 3.0).sum())

            # if args.dataset == 'nuscenes':
            #     gt_occ, save_info = read_occ3d_label(img_metas[0], nusc, real_idx)
            #     camera_mask = torch.from_numpy(gt_occ['mask_camera']).bool()#.cuda()
            #     lidar_mask = torch.from_numpy(gt_occ['mask_lidar']).bool()#.cuda()
            #     gt_vox = torch.from_numpy(gt_occ['semantics'])#.cuda()
            #     gt_vox[gt_vox == 17] = 0
            #     gt_vox = gt_vox.permute(1, 0, 2)
            #     camera_mask = camera_mask.permute(1, 0, 2)
            #     lidar_mask = lidar_mask.permute(1, 0, 2)
            # else:
            #     gt_vox, save_info = read_semantic_kitti(img_metas[0], real_idx)
            #     lidar_mask = gt_vox != 255
            # if not args.model_pred:
            #     predict_vox = gt_vox
            grid_size = predict_vox.shape

            # if args.use_lidar_mask:
            #     predict_vox[~lidar_mask] = 0
            # elif args.use_cam_mask and args.dataset == 'nuscenes':
            #     predict_vox[~camera_mask] = 0

            predict_vox[..., (grid_size[2] - args.cap):] = 0            
            if args.sem: 
                if args.model_pred:
                    if args.transform:
                        sem = result_dict['logits'].permute(3, 0, 1, 2) # C, H, W, D
                        sampled_sem = F.grid_sample(
                            sem[None, ...], # 1, C, H, W, D
                            lidar_points[..., [2, 0, 1]] * 2 - 1,
                            mode='bilinear',
                            align_corners=True) # 1, C, 200, 200, 16
                        sampled_sem = torch.argmax(sampled_sem, dim=1).squeeze()
                        sem = sampled_sem.permute(1, 0, 2)
                    else:
                        sem = result_dict['sem']
                    if args.dataset == 'nuscenes':
                        sem = openseed2nuscenes(sem)
                    elif args.dataset == 'kitti':
                        sem = cityscapes2semantickitti(sem)
                    else:
                        raise NotImplementedError
                    sem = sem.cpu()

                    predict_vox = predict_vox * sem
            else:
                for z in range(grid_size[2]-args.cap):
                    mask = (predict_vox > 0)[..., z]
                    predict_vox[..., z][mask] = z + 1 # grid_size[2] - z
        
            if args.dataset == 'nuscenes':
                # ego2lidar = img_metas[0]['ego2lidar']
                # ego2lidar = xyz.new_tensor(ego2lidar) # 4, 4
                # lidar_points = torch.matmul(ego2lidar.unsqueeze(0), xyz.reshape(-1, 4, 1))
                # lidar_points = lidar_points.squeeze(-1)[:, :3] # 200*200*16, 3
                # lidar_points = lidar_points[..., [1, 0, 2]]
                # gird_coords = lidar_points.cpu().numpy()
                grid_coords = result_dict['xyz'].reshape(-1, 3).cpu().numpy()
                voxel_origin = [-40.0, -40.0, -1.0]
                voxel_max = [40.0, 40.0, 5.4]
            else:
                voxel_origin = point_cloud_range[:3]
                voxel_max = point_cloud_range[3:]

            # visualization
            if args.save_rgb:
                # frame_dir = os.path.join(vis_save_path, str(real_idx))
                # os.makedirs(frame_dir, exist_ok=True)
                # gt_imgs = input_imgs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 256
                # gt_imgs = gt_imgs[..., [2, 1, 0]]
                # gt_imgs = gt_imgs.astype(np.uint8)
                for i in range(6):
                    # img = Image.fromarray(gt_imgs[i])
                    # img.save(osp.join(clip_dirs[i], 'img_'+str(real_idx)+'.jpg'))
                # if video_clip < 6:
                    filename = img_metas[0]['input_imgs_path'][i]
                    shutil.copy(filename, os.path.join(clip_dirs[i], 'img_'+str(real_idx)+'.jpg'))
            # save_path = os.path.join(vis_save_path, save_info)
            
            draw(predict_vox, 
                grid_coords, 
                [resolution] * 3, 
                clip_dirs,
                img_metas[0]['cam_positions'],
                img_metas[0]['focal_positions'],
                timestamp=real_idx,
                mode=args.mode,
                sem=args.sem,
                dataset=args.dataset)
            
            logger.info(f'done processing frame idx{idx}')
            
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)

    parser.add_argument('--dir-name', type=str, default='vis_scene')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0])
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')
    parser.add_argument('--cap', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model-pred', action='store_true', default=False)
    parser.add_argument('--scene-size', type=int, default=0)
    parser.add_argument('--sem', action='store_true', default=False)
    parser.add_argument('--use-lidar-mask', action='store_true', default=False)
    parser.add_argument('--use-cam-mask', action='store_true', default=False)
    parser.add_argument('--transform', action='store_true', default=False)
    parser.add_argument('--save-rgb', action='store_true', default=False)
    parser.add_argument('--scene-name', type=str, default='')
    parser.add_argument('--start-idx', type=int, default=0)
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
