<!-- TODO -->

# Visualization Guide

## Install packages

```
pip install pyvirtualdisplay mayavi PyQt5
```

## 3D Occupancy Prediction

```shell
# nuscenes
CUDA_VISIBLE_DEVICES=0 python vis_3d.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ --resume-from out/nuscenes/occ/model_state_dict.pth --frame-idx 0 100 200 --cap 4 --model-pred --scene-size 4 --sem --save-rgb

# semantickitti
CUDA_VISIBLE_DEVICES=0 python vis_3d.py --py-config config/kitti/kitti_occ.py --work-dir out/kitti/occ --resume-from out/kitti/occ/model_state_dict.pth --frame-idx 0 100 200 --cap 4 --dataset kitti  --model-pred --save-rgb
```

## 2D Visualization

```shell
CUDA_VISIBLE_DEVICES=0 python vis_pics.py --py-config config/nuscenes/nuscenes_novel_depth.py --work-dir out/nuscenes/novel_depth --resume-from out/nuscenes/novel_depth/model_state_dict.pth --vis-nerf-rgb ms_colors --frame-idx 0 100 200 --num-rays 96 200
```

## Video

```shell
# download the pkl file which includes lidar pointcloud when processing sweeps
cd data
wget --content-disposition https://cloud.tsinghua.edu.cn/f/e415321fb5f64c7081d2/?dl=1
cd ..

# generate individual images
CUDA_VISIBLE_DEVICES=0 python vis_3d_scene.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ --resume-from out/nuscenes/occ/model_state_dict.pth --cap 4 --model-pred --scene-size 4 --sem --save-rgb --scene-name scene-0268

# generate a video
python generate_videos.py --scene-dir out/nuscenes/occ/vis_scene/epoch9 --scene-name scene-0268 
```