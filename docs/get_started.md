<!-- template from bevformer -->

# Prerequisites

**Please ensure you have prepared the environment and datasets.**

[23/12/16 Update] Please update the timm package to 0.9.2 to run the training script.


# 3D Occupancy Prediction

## NuScenes

### Training 

```
python train.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ_train --depth-metric
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/831c104c82a244e9878a/) and put it under out/nuscenes/occ/
```
python eval_iou.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ --resume-from out/nuscenes/occ/model_state_dict.pth --occ3d --resolution 0.4 --sem --use-mask --scene-size 4
```

## SemanticKITTI

### Training

```
python train.py --py-config config/kitti/kitti_occ.py --work-dir out/kitti/occ_train --depth-metric --dataset kitti
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/3c09a5e8f5b94fa29289/) and put it under out/kitti/occ/
```
python eval_iou_kitti.py --py-config config/kitti/kitti_occ.py --work-dir out/kitti/occ --resume-from out/kitti/occ/model_state_dict.pth 
```

# Novel Depth Synthesis

## NuScenes

### Training

```
python train.py --py-config config/nuscenes/nuscenes_novel_depth.py --work-dir out/nuscenes/novel_depth_train --depth-metric
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/2d217cd298a34ed19039/) and put it under out/nuscenes/novel_depth/
```
python eval_novel_depth.py --py-config config/nuscenes/nuscenes_novel_depth.py --work-dir out/nuscenes/novel_depth --resume-from out/nuscenes/novel_depth/model_state_dict.pth
```

## SemanticKITTI

### Training 

```
python train.py --py-config config/kitti/kitti_novel_depth.py --work-dir out/kitti/novel_depth_train --depth-metric --dataset kitti
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/7280a44340fd440cba7c/) and put it under out/kitti/novel_depth/
```
python eval_novel_depth_kitti.py --py-config config/kitti/kitti_novel_depth.py --work-dir out/kitti/novel_depth --resume-from out/kitti/novel_depth/model_state_dict.pth 
```


# Depth Estimation

## nuScenes

### Training 

```
python train.py --py-config config/nuscenes/nuscenes_depth.py --work-dir out/nuscenes/depth_train --depth-metric
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/1a722b9139234542ae1e/) and put it under out/nuscenes/depth/
```
python eval_depth.py --py-config config/nuscenes/nuscenes_depth.py --work-dir out/nuscenes/depth --resume-from out/nuscenes/depth/model_state_dict.pth --depth-metric --batch 90000
```

Note that evaluating at a resolution (450\*800) of 1:2 against the raw image (900\*1600) takes about 90 min, because we batchify rays for rendering due to GPU memory limit. You can change the rendering resolution by the variable *NUM_RAYS* in utils/config_tools.py


## KITTI-2015

### Training

```
python train.py --py-config config/kitti_raw/kitti_raw_depth.py --work-dir out/kitti_raw/depth_train --depth-metric --dataset kitti
```

### Evaluation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/f87f6876569e4fdeb967/) and put it under out/kitti_raw/depth/
```
python eval_depth.py --py-config config/kitti_raw/kitti_raw_depth.py --work-dir out/kitti_raw/depth --resume-from out/kitti_raw/depth/model_state_dict.pth --depth-metric --dataset kitti_raw
```
