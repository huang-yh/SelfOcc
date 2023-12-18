# SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction
### [Paper](https://arxiv.org/pdf/2311.12754)  | [Project Page](https://huang-yh.github.io/SelfOcc/) 

> SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction

> [Yuanhui Huang](https://scholar.google.com/citations?hl=zh-CN&user=LKVgsk4AAAAJ)*, [Wenzhao Zheng](https://wzzheng.net/)\* $\dagger$, [Borui Zhang](https://boruizhang.site/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)$\ddagger$

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author

SelfOcc empowers 3D autonomous driving world models (e.g., [OccWorld](https://github.com/wzzheng/OccWorld)) with scalable 3D representations, paving the way for **interpretable end-to-end large driving models**.

## News
- **[2023/12/16]** Training code release.
- **[2023/11/28]** Evaluation code release.
- **[2023/11/20]** Paper released on [arXiv](https://arxiv.org/abs/2311.12754).
- **[2023/11/20]** Demo release.

## Demo

### Trained using only video sequences and poses:

![demo](./assets/iou.gif)

### Trained using an additional off-the-shelf 2D segmentor ([OpenSeeD](https://github.com/IDEA-Research/OpenSeeD)):

![demo](./assets/miou.gif)

![legend](./assets/legend.png)


### More demo videos can be downloaded [here](https://cloud.tsinghua.edu.cn/d/640283b528f7436193a4/).

## Overview
![overview](./assets/overview.png)

- We first transform the images into the 3D space (e.g., bird's eye view, [tri-perspective view](https://github.com/wzzheng/TPVFormer)) to obtain 3D representation of the scene. We directly impose constraints on the 3D representations by treating them as signed distance fields. We can then render 2D images of previous and future frames as self-supervision signals to learn the 3D representations. 

- Our SelfOcc outperforms the previous best method SceneRF by 58.7% using a single frame as input on SemanticKITTI and is the first self-supervised work that produces reasonable 3D occupancy for surround cameras on nuScenes. 

- SelfOcc produces high-quality depth and achieves state-of-the-art results on **novel depth synthesis**, **monocular depth estimation**, and **surround-view depth estimation** on the SemanticKITTI, KITTI-2015, and nuScenes, respectively. 

## Results
<img src=./assets/results.png height="480px" width="563px" />

## Getting Started

### Installation

Follow detailed instructions in [Installation](docs/installation.md).

### Preparing Dataset

Follow detailed instructions in [Prepare Dataset](docs/prepare_data.md).

### Run

[23/12/16 Update] Please update the timm package to 0.9.2 to run the training script.

#### 3D Occupancy Prediction

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/831c104c82a244e9878a/) and put it under out/nuscenes/occ/
```bash
# train
python train.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ_train --depth-metric
# eval
python eval_iou.py --py-config config/nuscenes/nuscenes_occ.py --work-dir out/nuscenes/occ --resume-from out/nuscenes/occ/model_state_dict.pth --occ3d --resolution 0.4 --sem --use-mask --scene-size 4
```

#### Novel Depth Synthesis


Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/2d217cd298a34ed19039/) and put it under out/nuscenes/novel_depth/
```bash
# train
python train.py --py-config config/nuscenes/nuscenes_novel_depth.py --work-dir out/nuscenes/novel_depth_train --depth-metric
# evak
python eval_novel_depth.py --py-config config/nuscenes/nuscenes_novel_depth.py --work-dir out/nuscenes/novel_depth --resume-from out/nuscenes/novel_depth/model_state_dict.pth
```

#### Depth Estimation

Download model weights [HERE](https://cloud.tsinghua.edu.cn/f/1a722b9139234542ae1e/) and put it under out/nuscenes/depth/
```bash
# train
python train.py --py-config config/nuscenes/nuscenes_depth.py --work-dir out/nuscenes/depth_train --depth-metric
# eval
python eval_depth.py --py-config config/nuscenes/nuscenes_depth.py --work-dir out/nuscenes/depth --resume-from out/nuscenes/depth/model_state_dict.pth --depth-metric --batch 90000
```

Note that evaluating at a resolution (450\*800) of 1:2 against the raw image (900\*1600) takes about 90 min, because we batchify rays for rendering due to GPU memory limit. You can change the rendering resolution by the variable *NUM_RAYS* in utils/config_tools.py

More details on more datasets are detailed in  [Run and Eval](docs/get_started.md).

## Related Projects

Our code is based on [TPVFormer](https://github.com/wzzheng/TPVFormer) and [PointOcc](https://github.com/wzzheng/PointOcc). 

Also thanks to these excellent open-sourced repos:
[SurroundOcc](https://github.com/weiyithu/SurroundOcc) 
[OccFormer](https://github.com/zhangyp15/OccFormer)
[BEVFormer](https://github.com/fundamentalvision/BEVFormer)

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{huang2023self,
    title={SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction},
    author={Huang, Yuanhui and Zheng, Wenzhao and Zhang, Borui and Zhou, Jie and Lu, Jiwen },
    journal={arXiv preprint arXiv:2311.12754},
    year={2023}
}
```
