import glob, os, time, random

import numpy as np
import torch
# from torchvision import transforms
from dataset.kitti.helpers import dump_xyz, read_calib, read_poses, read_rgb
from dataset.kitti.params import val_error_frames
# from . import io_data as SemanticKittiIO
from copy import deepcopy
from mmcv.image.io import imread
from .. import OPENOCC_DATASET

if 'HFAI' in os.environ:
    hfai = True
    from dataset.loading import LoadMultiViewImageFromFilesHF, LoadPtsFromFilesHF
else:
    hfai = False


@OPENOCC_DATASET.register_module()
class Kitti_One_Frame:
    def __init__(
            self,
            split,
            root, preprocess_root,
            frames_interval=0.4,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            sequences=None,
            selected_frames=None, 
            n_rays=1200,
            cur_prob=1.0,
            crop_size=[370, 1220],
            strict=True,
            return_depth=False,
            prev_prob=0.5,
            choose_nearest=False,
            return_sem=False,
            sem_path=None,
            **kwargs,
    ):
        self.root = root
        self.preprocess_root = preprocess_root
        self.depth_preprocess_root = os.path.join(preprocess_root, "depth")
        self.transform_preprocess_root = os.path.join(preprocess_root, "transform")
        self.n_classes = 20
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        self.n_rays = n_rays
        self.cur_prob = cur_prob
        self.return_depth = return_depth
        self.prev_prob = prev_prob
        self.choose_nearest = choose_nearest
        self.return_sem = return_sem
        assert (not return_sem) or os.path.exists(sem_path)
        self.sem_path = sem_path
        
        self.transxy = [
            [0, -1., 0, 0],
            [1., 0, 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]]
        self.transxy = np.array(self.transxy)

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],                   
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split
        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = splits[split]
        self.output_scale = 1
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.frames_interval = frames_interval
        if not isinstance(sequence_distance, list):
            sequence_distance = [sequence_distance] * 2
        self.sequence_distance = sequence_distance
        self.strict = strict

        self.voxel_size = 0.2  # 0.2m
        self.img_W = crop_size[1]
        self.img_H = crop_size[0]

      
        start_time = time.time()
        self.scans = []
        self.frame2scan = {}

        for sequence in self.sequences:
            pose_path = os.path.join(self.root, "dataset", "poses", sequence + ".txt")
            gt_global_poses = read_poses(pose_path)

            calib = read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]

            T_cam0_2_cam2 = calib['T_cam0_2_cam2']
            T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
            T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]

            if split == "val":
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
                )

            else:
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", "*.png"
                )

            
            seq_img_paths = glob.glob(glob_path)

            max_length = 0
            min_length = 50
            paired_dists = {}
            dist_step = 1 if split == 'train' else 5
            for seq_img_path in seq_img_paths:
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]
                prev_frame_id = "{:06d}".format(int(frame_id) - dist_step)
                img_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", prev_frame_id + ".png")
                if not os.path.exists(img_path):
                    paired_dists[frame_id] = 0.
                else:
                    curr_pose = gt_global_poses[int(frame_id)]
                    prev_pose = gt_global_poses[int(prev_frame_id)]
                    prev_xyz = dump_xyz(prev_pose)
                    curr_xyz = dump_xyz(curr_pose)
                    rel_distance = np.sqrt(
                        (prev_xyz[0] - curr_xyz[0]) ** 2 + (prev_xyz[2] - curr_xyz[2]) ** 2)
                    paired_dists[frame_id] = rel_distance
            
            for seq_img_path in seq_img_paths:
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]

                current_img_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png")
                current_lid_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "velodyne", frame_id + ".bin")

                prev_frame_ids, next_frame_ids = [], []
                prev_img_paths, next_img_paths = [], []
                prev_lid_paths, next_lid_paths = [], []
                prev_poses, next_poses = [], []
                prev_dists, next_dists = [], []

                if self.split == 'train':
                    pos_step = 1
                    neg_step = -1
                else:
                    pos_step = 5
                    neg_step = -5
                # deal with prev
                step = -1
                cnt = 0
                dist = 0.
                while True:
                    # cnt += step
                    cnt += neg_step
                    rel_frame_id = "{:06d}".format(int(frame_id) + cnt)

                    img_path = os.path.join(
                        self.root, "dataset", "sequences", sequence, "image_2", rel_frame_id + ".png")

                    if not os.path.exists(img_path):
                        break

                    dist += paired_dists["{:06d}".format(int(rel_frame_id) + pos_step)]
                    if dist < frames_interval:
                        continue
                    if dist > sequence_distance[0]:
                        break
                    if split == "val" and rel_frame_id in val_error_frames:
                        continue

                    prev_frame_ids.append(rel_frame_id)
                    prev_img_paths.append(img_path)
                    lidar_path = os.path.join(
                        self.root, "dataset", "sequences", sequence, "velodyne", rel_frame_id + ".bin")
                    prev_lid_paths.append(lidar_path)
                    prev_poses.append(gt_global_poses[int(rel_frame_id)])
                    prev_dists.append(dist)

                # deal with next
                step = 1
                cnt = 0
                dist = 0.
                while True:
                    # cnt += step
                    cnt += pos_step
                    rel_frame_id = "{:06d}".format(int(frame_id) + cnt)

                    img_path = os.path.join(
                        self.root, "dataset", "sequences", sequence, "image_2", rel_frame_id + ".png")

                    if not os.path.exists(img_path):
                        break

                    dist += paired_dists[rel_frame_id]
                    if dist < frames_interval:
                        continue
                    if dist > sequence_distance[1]:
                        break
                    if split == "val" and rel_frame_id in val_error_frames:
                        continue

                    next_frame_ids.append(rel_frame_id)
                    next_img_paths.append(img_path)
                    lidar_path = os.path.join(
                        self.root, "dataset", "sequences", sequence, "velodyne", rel_frame_id + ".bin")
                    next_lid_paths.append(lidar_path)
                    next_poses.append(gt_global_poses[int(rel_frame_id)])
                    next_dists.append(dist)

                # ignore error frame
                if split == "val" and frame_id in val_error_frames:
                    continue

                is_included = True if selected_frames is None else frame_id in selected_frames

                if is_included:

                    prev_next_len = len(prev_poses) + len(next_poses)
                    
                    if prev_next_len > max_length:
                        max_length = prev_next_len
                    if prev_next_len < min_length:
                        min_length = prev_next_len

                    self.frame2scan.update({str(sequence) + "_" + frame_id: len(self.scans)})

                    if not self.strict:
                        prev_img_paths.append(current_img_path)
                        prev_lid_paths.append(current_lid_path)
                        next_img_paths.append(current_img_path)
                        next_lid_paths.append(current_lid_path)
                        prev_poses.append(curr_pose)
                        next_poses.append(curr_pose)
                        prev_dists.append(0.)
                        next_dists.append(0.)
                        prev_frame_ids.append(frame_id)
                        next_frame_ids.append(frame_id)

                    self.scans.append(
                        {
                            "frame_id": frame_id,
                            "sequence": sequence,
                            "img_path": current_img_path,
                            "lid_path": current_lid_path,
                            "pose": gt_global_poses[int(frame_id)],

                            "prev_img_paths": prev_img_paths,
                            "prev_lid_paths": prev_lid_paths,
                            "next_img_paths": next_img_paths,
                            "next_lid_paths": next_lid_paths,

                            "T_velo_2_cam": T_velo_2_cam,
                            "P": P,
                            "T_cam0_2_cam2": T_cam0_2_cam2,
                            "T_cam2_2_cam0": T_cam2_2_cam0,

                            "prev_poses": prev_poses,
                            "next_poses": next_poses,
                            "prev_dists": prev_dists,
                            "next_dists": next_dists,
                            "prev_frame_ids": prev_frame_ids,
                            "next_frame_ids": next_frame_ids
                        }
                    )
            print(sequence, min_length, max_length)

        # self.to_tensor_normalized = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        # self.to_tensor = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size):
        # lidar2img: N, 4, 4
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(self.transxy @ scan.T, 0) # 1, 4, n
        img_points = np.transpose(lidar2img @ points_hcoords, (0, 2, 1)) # N, n, 4

        depth = img_points[..., 2] # N, n
        img_points = img_points[..., :2] # N, n, 2
        mask = (depth < self.eval_depth) & (depth > 1e-3)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        img_points[..., 0] = img_points[..., 0] / image_size[1]
        img_points[..., 1] = img_points[..., 1] / image_size[0]
        # img_points = np.round(img_points).astype(int)
        mask = mask & (img_points[..., 0] > 0) & \
                    (img_points[..., 1] > 0) & \
                    (img_points[..., 0] < 1) & \
                    (img_points[..., 1] < 1)

        return img_points, depth, mask
    
    def load_2d_sem_label(self, scan):
        sequence = scan['sequence']
        filename = os.path.basename(scan['img_path'])
        sem_path = os.path.join(self.sem_path, sequence, 'image_02', filename + '.npy')
        sem = np.load(sem_path)[None, ...]
        return sem
    
    def __len__(self):
        return len(self.scans)
    
    def prepare_img_metas(self, scan, anchor_scan, anchor_prev, anchor_next):
        img_metas = {}
        img_metas.update({
            'input_imgs_path': [scan['img_path']],
            'curr_imgs_path': [anchor_scan['img_path']],
            'prev_imgs_path': [anchor_scan['prev_img_paths'][anchor_prev]],
            'next_imgs_path': [anchor_scan['next_img_paths'][anchor_next]]
        })

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = scan['P'][:3, :3]
        lidar2img = intrinsic @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        img2lidar = np.linalg.inv(lidar2img)

        temImg2lidar = self.transxy @ np.linalg.inv(scan['T_velo_2_cam']) @ \
            scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(scan['pose']) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        img2prevImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['prev_poses'][anchor_prev]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        img2nextImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['next_poses'][anchor_next]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        img_metas.update({
            'lidar2img': np.expand_dims(lidar2img, axis=0),
            'img2lidar': [img2lidar],
            'img2prevImg': [img2prevImg],
            'img2nextImg': [img2nextImg],
            'token': scan['frame_id'],
            'sequence': scan['sequence'],
            'temImg2lidar': [temImg2lidar]
        })

        return img_metas

    def read_surround_imgs(self, img_paths):
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32))
        imgs = [img[:self.img_H, :self.img_W, :] for img in imgs]
        return imgs

    def __getitem__(self, index):
        #### 1. get color, temporal_depth choice if necessary
        if random.random() < self.cur_prob:
            temporal_supervision = 'curr'
        elif random.random() < self.prev_prob:
            temporal_supervision = 'prev'
        else:
            temporal_supervision = 'next'

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        while True:
            scan = deepcopy(self.scans[index])
            sequence = scan['sequence']
            # scan_id = scan['frame_id']
            # day, seq = scan['sequence']
            # is_right = scan['is_right']
            # stereo_sign = 'r' if is_right else 'l'

            if temporal_supervision == 'curr':
                anchor_scan = deepcopy(scan)
            elif temporal_supervision == 'prev':
                if len(scan['prev_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan_id = np.random.choice(scan['prev_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[str(sequence) + '_' + anchor_scan_id]])
            else:
                if len(scan['next_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan_id = np.random.choice(scan['next_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[str(sequence) + '_' + anchor_scan_id]])
                            
            if len(anchor_scan['prev_frame_ids']) == 0 or len(anchor_scan['next_frame_ids']) == 0:
                index = np.random.randint(len(self))
                continue
            anchor_prev = 0 if self.choose_nearest else np.random.randint(len(anchor_scan['prev_frame_ids']))
            anchor_next = 0 if self.choose_nearest else np.random.randint(len(anchor_scan['next_frame_ids']))
            break

        img_metas = self.prepare_img_metas(
            scan, anchor_scan, anchor_prev, anchor_next)
        
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                scan['lid_path'], img_metas['lidar2img'], [self.img_H, self.img_W])
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})
        if self.return_sem:
            sem = self.load_2d_sem_label(anchor_scan)
            img_metas.update({'sem': sem})

        # read 6 cams
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'])
        curr_imgs = self.read_surround_imgs(img_metas['curr_imgs_path'])
        prev_imgs = self.read_surround_imgs(img_metas['prev_imgs_path'])
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'])

        data_tuple = ([input_imgs, curr_imgs, prev_imgs, next_imgs], img_metas)
        return data_tuple   


if __name__ == "__main__":
    import os, sys
    print(sys.path)

    dataset = Kitti_Smart(
        'train',
        root='data/kitti',
        preprocess_root='data/kitti/preprocess',
        cur_prob=1.0,
        crop_size=[352, 1216])
    
    batch = dataset[0]
    
    import pdb; pdb.set_trace()