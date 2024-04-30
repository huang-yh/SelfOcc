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


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
    dataset='nuscenes',
    save_path=None
):
    w, h, z = voxels.shape
    # grid = grid.astype(np.int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
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
            # fov_voxels[:, 1],
            # fov_voxels[:, 0],
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            # colormap="hot",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            # fov_voxels[:, 1],
            # fov_voxels[:, 0],
            fov_voxels[:, 0],
            fov_voxels[:, 1],
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
                ]
            ).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    if dataset == 'nuscenes':
        scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(-5)
        scene.camera.orthogonalize_view_up()
        scene.render()
    else:
        # camera view
        scene.camera.position = cam_positions[0] # - np.array([0.7, 1.3, 0.])
        scene.camera.focal_point = focal_positions[0] # - np.array([0.7, 1.3, 0.])
        scene.camera.view_angle = 41
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        scene.camera.compute_view_plane_normal()
        scene.render()

    if offscreen:
        mlab.savefig(save_path)
    else:
        mlab.show()
    mlab.close()


def main(local_rank, args):
    if args.dataset == 'nuscenes':
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
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
            'type': 'nuScenes_One_Frame',
            'imageset': 'data/nuscenes_infos_val_temporal_v1_scene_1.pkl',
            'scene_name': args.scene_name,
            'strict': False})
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
    os.makedirs(vis_save_path, exist_ok=True)

    with torch.no_grad():
        num_iters = len(val_dataset_loader) if args.scene_name else len(args.frame_idx)
        if args.scene_name:
            loader_iter = iter(val_dataset_loader)
        for idx in range(num_iters):
            if args.scene_name:
                input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, img_metas, _, _, _ = next(loader_iter)
                real_idx = img_metas[0]['timestamp']
            else:
                real_idx = args.frame_idx[idx]
                one_batch = dataset[real_idx]
                input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, img_metas, _, _, _ = custom_collate_fn_temporal([one_batch])

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
                    resolution = 0.3
                elif args.scene_size == 4:
                    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
                    resolution = 0.4

            if args.model_pred:
                if cfg.get('estimate_pose'):
                    assert curr_aug is not None and prev_aug is not None and next_aug is not None
                    assert img_metas[0]['input_imgs_path'] == img_metas[0]['curr_imgs_path']
                    curr_aug, prev_aug, next_aug = curr_aug.cuda(), prev_aug.cuda(), next_aug.cuda()
                    pose_dict = my_model(pose_input=[curr_aug, prev_aug, next_aug], metas=img_metas, predict_pose=True)
                    for i_meta, meta in enumerate(img_metas):
                        meta.update(pose_dict[i_meta])

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

            if args.dataset == 'nuscenes':
                gt_occ, save_info = read_occ3d_label(img_metas[0], nusc, real_idx)
                camera_mask = torch.from_numpy(gt_occ['mask_camera']).bool()#.cuda()
                lidar_mask = torch.from_numpy(gt_occ['mask_lidar']).bool()#.cuda()
                gt_vox = torch.from_numpy(gt_occ['semantics'])#.cuda()
                gt_vox[gt_vox == 17] = 0
                gt_vox = gt_vox.permute(1, 0, 2)
                camera_mask = camera_mask.permute(1, 0, 2)
                lidar_mask = lidar_mask.permute(1, 0, 2)
            else:
                gt_vox, save_info = read_semantic_kitti(img_metas[0], real_idx)
                lidar_mask = gt_vox != 255
            if not args.model_pred:
                predict_vox = gt_vox
            grid_size = predict_vox.shape

            if args.use_lidar_mask:
                predict_vox[~lidar_mask] = 0
            elif args.use_cam_mask and args.dataset == 'nuscenes':
                predict_vox[~camera_mask] = 0

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
                voxel_origin = [-40.0, -40.0, -1.0]
                voxel_max = [40.0, 40.0, 5.4]
                resolution = 0.4
            else:
                voxel_origin = point_cloud_range[:3]
                voxel_max = point_cloud_range[3:]

            # visualization
            if args.save_rgb:
                frame_dir = os.path.join(vis_save_path, str(real_idx))
                os.makedirs(frame_dir, exist_ok=True)
                gt_imgs = curr_imgs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 256
                gt_imgs = gt_imgs[..., [2, 1, 0]]
                gt_imgs = gt_imgs.astype(np.uint8)
                for i in range(gt_imgs.shape[0]):
                    img = Image.fromarray(gt_imgs[i])
                    img.save(osp.join(frame_dir, f'gtimg{i}.png'))
            save_path = os.path.join(vis_save_path, save_info)
            
            draw(predict_vox, 
                None, # predict_pts,
                voxel_origin, 
                [resolution] * 3, 
                None, # grid.squeeze(0).cpu().numpy(), 
                None, # pt_label.squeeze(-1),
                None,
                img_metas[0]['cam_positions'] if 'cam_positions' in img_metas[0] else None,
                img_metas[0]['focal_positions'] if 'focal_positions' in img_metas[0] else None,
                timestamp=timestamp,
                mode=args.mode,
                sem=args.sem,
                dataset=args.dataset,
                save_path=save_path)
            
            logger.info(f'done processing frame idx{idx}')
            
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)

    parser.add_argument('--dir-name', type=str, default='vis_3d')
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
