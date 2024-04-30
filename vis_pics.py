import os
import time, argparse, math, os.path as osp, numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')
from pathlib import Path
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

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


def pass_print(*args, **kwargs):
    pass

def create_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((19, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]   # road
    colormap[1] = [244, 35, 232]   # 
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap

def save_depth_map(disp_resized_np, save_path):
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save(save_path)
    return

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, args.dataset, False, args)
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
    logger = MMLogger('selfocc', log_file=log_file)

    # build model    
    cfg.model.head.ray_sample_mode = 'fixed'
    cfg.model.head.return_second_grad = False
    cfg.model.head.return_max_depth = True
    if len(args.novel_view) > 0:
        cfg.model.head.novel_view = args.novel_view
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
    dataset = val_dataset_loader.dataset
    print('length of dataset: ', len(dataset))
    
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
        # revise ckpt
        if 'head.rays' in state_dict:
            del state_dict['head.rays']
        print(raw_model.load_state_dict(state_dict, strict=False))
        print(f'successfully resumed from {cfg.resume_from}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()
    amp = cfg.get('amp', False)

    with torch.no_grad():
        for idx in args.frame_idx:
            vis_save_path = osp.join(args.work_dir, args.dir_name, f'epoch{epoch}', f'{idx}')
            os.makedirs(vis_save_path, exist_ok=True)

            one_batch = dataset[idx]
            input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, img_metas, curr_aug, prev_aug, next_aug = \
                custom_collate_fn_temporal([one_batch])
            input_imgs = input_imgs.to(args.device)

            with torch.cuda.amp.autocast(amp):
                if cfg.get('estimate_pose'):
                    assert curr_aug is not None and prev_aug is not None and next_aug is not None
                    assert img_metas[0]['input_imgs_path'] == img_metas[0]['curr_imgs_path']
                    curr_aug, prev_aug, next_aug = curr_aug.cuda(), prev_aug.cuda(), next_aug.cuda()
                    pose_dict = my_model(pose_input=[curr_aug, prev_aug, next_aug], metas=img_metas, predict_pose=True)
                    for i_meta, meta in enumerate(img_metas):
                        meta.update(pose_dict[i_meta])

                my_model(imgs=input_imgs, metas=img_metas, prepare=True)
                if distributed:
                    result_dict = my_model.module.head.render(metas=img_metas, batch=args.batch)
                else:
                    result_dict = my_model.head.render(metas=img_metas, batch=args.batch)

            # visualization
            gt_imgs = curr_imgs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            gt_imgs = (gt_imgs * 256).astype(np.uint8)
            gt_imgs = gt_imgs[..., [2, 1, 0]]
            for i in range(gt_imgs.shape[0]):
                gt_imgs_tmp = pil.fromarray(gt_imgs[i])
                gt_imgs_tmp.save(osp.join(vis_save_path, f'i{idx}_gtimg{i}.png'))
                plt.imshow(gt_imgs[i])
                plt.savefig(osp.join(vis_save_path, f'i{idx}_gtimg{i}.png'))
                        
            if args.vis_2d_depth:
                to_vis = result_dict[args.vis_2d_depth]
                if not isinstance(to_vis, list):
                    to_vis = [to_vis]
                for scale, to_vis_i in enumerate(to_vis):
                    to_vis_i = to_vis_i.cpu().squeeze()
                    num_cams = to_vis_i.shape[0]
                    for i in range(num_cams):
                        plt.imshow(to_vis_i[i])
                        plt.savefig(osp.join(
                            vis_save_path, f'i{idx}_{args.vis_2d_depth}{i}_s{scale}.png'))
            
            if len(args.vis_nerf_depth) > 0:
                for curr_kw in args.vis_nerf_depth:
                    to_vis = result_dict[curr_kw]
                    if not isinstance(to_vis, list):
                        to_vis = [to_vis]
                    for scale, to_vis_i in enumerate(to_vis):
                        to_vis_i = to_vis_i.cpu().squeeze(0)
                        num_cams = to_vis_i.shape[0]
                        for i in range(num_cams):
                            to_vis_tmp = to_vis_i[i].reshape(*cfg.num_rays).numpy()
                            to_vis_tmp = 80.0 / to_vis_tmp 
                            save_depth_map(
                                to_vis_tmp, 
                                osp.join(vis_save_path, f'i{idx}_{curr_kw}{i}_s{scale}.png'))
            
            if args.vis_nerf_rgb:
                rgb_gt_imgs = color_imgs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
                rgb_gt_imgs = (rgb_gt_imgs * 256).astype(np.uint8)
                rgb_gt_imgs = rgb_gt_imgs[..., [2, 1, 0]]
                for i in range(rgb_gt_imgs.shape[0]):
                    rgb_gt_img = pil.fromarray(rgb_gt_imgs[i])
                    rgb_gt_img.save(osp.join(vis_save_path, f'i{idx}_rgb_gtimg{i}.png'))

                to_vis = result_dict[args.vis_nerf_rgb]
                if not isinstance(to_vis, list):
                    to_vis = [to_vis]
                for scale, to_vis_i in enumerate(to_vis):
                    to_vis_i = to_vis_i.cpu().squeeze(0)
                    num_cams = to_vis_i.shape[0]
                    to_vis_i = torch.clamp(to_vis_i, 0., 1.) * 255
                    to_vis_i = to_vis_i.numpy().astype(np.uint8)
                    to_vis_i = to_vis_i[..., [2, 1, 0]]
                    to_vis_i = to_vis_i.reshape(num_cams, *cfg.num_rays, 3)
                    for i in range(num_cams):
                        to_vis_i_tmp = pil.fromarray(to_vis_i[i])
                        to_vis_i_tmp.save(osp.join(
                            vis_save_path, f'i{idx}_{args.vis_nerf_rgb}{i}_s{scale}.png'))
            
            if args.vis_nerf_sem:
                colormap = create_label_colormap()
                to_vis = result_dict[args.vis_nerf_sem]
                if not isinstance(to_vis, list):
                    to_vis = [to_vis]
                for scale, to_vis_i in enumerate(to_vis):
                    to_vis_i = torch.argmax(to_vis_i, dim=-1) # B, N, R
                    to_vis_i = to_vis_i.cpu().squeeze(0).numpy() # N, R
                    num_cams = to_vis_i.shape[0]
                    to_vis_i = colormap[to_vis_i]
                    for i in range(num_cams):
                        plt.imshow(to_vis_i[i].reshape(*cfg.num_rays, 3))
                        plt.savefig(osp.join(
                            vis_save_path, f'i{idx}_{args.vis_nerf_sem}{i}_s{scale}.png'))
            
            logger.info(f'done processing frame idx{idx}')
            
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)

    parser.add_argument('--vis-2d-depth', type=str, default='')
    parser.add_argument('--vis-nerf-depth', type=str, default=['ms_depths', 'ms_max_depths'], nargs='+')
    parser.add_argument('--vis-nerf-rgb', type=str, default='')
    parser.add_argument('--vis-nerf-sem', type=str, default='')

    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0, 100, 300, 200, 500])
    parser.add_argument('--vis-train', action='store_true', default=False)

    parser.add_argument('--novel-view', type=float, nargs='+', default=[])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--num-rays', type=int, nargs='+', default=[24, 50])
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
