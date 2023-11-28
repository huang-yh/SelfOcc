<!-- template from bevformer -->

## NuScenes

**a. Download nuScenes data.**

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

**b. Download occupancy annotations.**

Download the gts.tar.gz of the trainval split of Occ3D-nuScenes [HERE](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction/tree/main#download) and unzip it.


**b. Download nuScenes pkl files.**

As a self-supervised method, we also include sweep data during training.
```shell
# pkl for keyframes and sweep data
wget --content-disposition https://cloud.tsinghua.edu.cn/f/ff15569b9e4d4086857e/?dl=1
wget --content-disposition https://cloud.tsinghua.edu.cn/f/fbe2ad8507494953abbe/?dl=1
# pkl for keyframes
wget --content-disposition https://cloud.tsinghua.edu.cn/f/07323a7a1c894e768924/?dl=1
```

**Folder structure**
```
SelfOcc
├── ...
├── data/
│   ├── nuscenes_infos_train_sweeps.pkl
│   ├── nuscenes_infos_val_sweeps.pkl
│   ├── nuscenes_infos_val_temporal_v2.pkl
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
│   ├── occ3d/
│   │   ├── gts/
│   │   |   ├── scene-0001/
│   │   |   ├── ...
```

## SemanticKITTI

We follow similar instructions as [SceneRF](https://github.com/astra-vision/SceneRF) to prepare SemanticKITTI.

**a. Download calib, rgb, pose and lidar files.**

To train and evaluate novel depths synthesis, please download on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) the following data:

    - Odometry data set (calibration files, 1 MB)
    - Odometry data set (color, 65 GB)
    - Odometry ground truth poses (4 MB)
    - Velodyne laser data, 80 GB

**b. Download occupancy annotations.**

To evaluate 3D occupancy prediction, please download **the SemanticKITTI voxel data (700 MB)** on [Semantic KITTI download website](http://www.semantic-kitti.org/dataset.html).

**c. Create preprocess folder.**

Create an empty folder to store preprocess data at `data/kitti/preprocess`.

**Folder structure**
```
SelfOcc
├── ...
├── data/
│   ├── kitti/
│   │   ├── dataset/
|   |   |   ├── poses/
|   |   |   |   ├── 00.txt
|   |   |   |   ├── ...
|   |   |   ├── sequences/
|   |   |   |   ├── 00/
|   |   |   |   |   ├── image_2/
|   |   |   |   |   ├── image_3/
|   |   |   |   |   ├── labels/
|   |   |   |   |   ├── velodyne/
|   |   |   |   |   ├── voxels/
|   |   |   |   |   ├── calib.txt
|   |   |   |   |   ├── poses.txt
|   |   |   |   |   ├── times.txt
|   |   |   |   ├── ...
│   │   ├── preprocess/
```

## KITTI-2015

We follow similar instructions as [BehindTheScenes](https://github.com/Brummi/BehindTheScenes) to prepare KITTI-2015.

To download KITTI, go to https://www.cvlibs.net/datasets/kitti/raw_data.php and create an account. We require all synched+rectified data, as well as the calibrations (using in fact only frames with ego motion). The website also provides scripts for automatic downloading of the different sequences. Following BehindTheScenes, we use the same poses computed from ORB-SLAM3 (can be found under dataset/kitti_raw/orb-slam_poses).

**Folder structure**
```
SelfOcc
├── ...
├── data/
│   ├── kitti_raw/
│   │   ├── 2011_09_26/
|   |   |   ├── 2011_09_26_drive_0001_sync/
|   |   |   |   ├── image_00/
|   |   |   |   ├── image_01/
|   |   |   |   ├── image_02/
|   |   |   |   ├── image_03/
|   |   |   |   ├── oxts/
|   |   |   |   ├── velodyne_points/
|   |   |   ├── ...
|   |   |   ├── calib_cam_to_cam.txt
|   |   |   ├── calib_imu_to_velo.txt
|   |   |   ├── calib_velo_to_cam.txt
│   │   ├── 2011_09_28/
│   │   ├── 2011_09_29/
│   │   ├── 2011_09_30/
│   │   ├── 2011_10_03/
```
