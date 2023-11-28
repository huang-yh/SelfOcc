import os, time, random
from pathlib import Path
from copy import deepcopy
from collections import Counter

import numpy as np
from mmcv.image.io import imread
from .. import OPENOCC_DATASET

BASE_SIZES = {
    "2011_09_26": (375, 1242),
    "2011_09_28": (370, 1224),
    "2011_09_29": (374, 1238),
    "2011_09_30": (370, 1226),
    "2011_10_03": (376, 1241),
}


@OPENOCC_DATASET.register_module()
class Kitti_Raw_Stereo:
    """
    Only support current timestamp supervision.
    """
    def __init__(
            self,
            root,
            pose_path, split_path,
            frames_interval=0.4,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            eigen_depth=True,
            cur_prob=1.0,
            crop_size=[370, 1220],
            strict=True,
            return_depth=False,
            prev_prob=0.5,
            choose_nearest=False,
            include_stereo=False,
            **kwargs,
    ):
        self.root = root
        self.pose_path = pose_path
        self.split_path = split_path
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        self.eigen_depth = eigen_depth
        self.cur_prob = cur_prob
        assert cur_prob == 1.
        self.return_depth = return_depth
        self.prev_prob = prev_prob
        self.frames_interval = frames_interval
        self.sequence_distance = sequence_distance
        self.strict = strict
        self.choose_nearest = choose_nearest
        self.include_stereo = include_stereo
        self.img_W = crop_size[1]
        self.img_H = crop_size[0]

        self.transxy = [
            [0, -1., 0, 0],
            [1., 0, 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]]
        self.transxy = np.array(self.transxy)      

        self._sequences = self._get_sequences(self.root)
        self._seq_lengths = {(day, seq): length for day, seq, length in self._sequences}

        self._calibs = self._load_calibs(self.root)
        self._poses = self._load_poses(self.pose_path, self._sequences)
        self._old_poses = self._load_poses('dataset/kitti_raw/orb-slam_poses', self._sequences)

        self._datapoints = self._load_split(self.split_path)
        self.length = len(self._datapoints)

        start_time = time.time()
        self.scans = []
        self.frame2scan = {}
        max_length = 0
        min_length = 50

        # for i_sample, sample in enumerate(self._datapoints):
        for day, seq, length in self._sequences:
            if len(self._old_poses[(day, seq)]) == 0:
                continue
            for frame_id in range(length):
                for is_right in [False, True]:

                    image_folder = "image_03" if is_right else "image_02"

                    current_img_path = os.path.join(self.root, day, seq, image_folder, "data", f"{frame_id:010d}.png")
                    current_lid_path = os.path.join(self.root, day, seq, "velodyne_points", "data", f"{frame_id:010d}.bin")
                    assert os.path.exists(current_img_path), current_img_path

                    curr_pose = self._poses[(day, seq)][frame_id, :, :]
                    curr_xyz = curr_pose[:3, 3]

                    calib = self._calibs[day]
                    P = calib["P_2"] if not is_right else calib["P_3"]
                    T_cam0_2_cam2 = calib['T_cam0_2_cam2'] if not is_right else calib['T_cam0_2_cam3']
                    T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
                    T_cam0_2_cam3 = calib['T_cam0_2_cam2'] if is_right else calib['T_cam0_2_cam3']
                    T_velo_2_img = calib['P_v2cr'] if is_right else calib['P_v2cl']

                    prev_frame_ids, next_frame_ids = [], []
                    prev_img_paths, next_img_paths = [], []
                    prev_lid_paths, next_lid_paths = [], []
                    prev_poses, next_poses = [], []
                    prev_dists, next_dists = [], []

                    pos_step = 1
                    neg_step = -1

                    # deal with prev
                    cnt = 0
                    while True:
                        cnt += neg_step
                        rel_frame_id = frame_id + cnt

                        img_path = os.path.join(self.root, day, seq, image_folder, "data", f"{rel_frame_id:010d}.png")

                        if not os.path.exists(img_path):
                            break

                        temp_pose = self._poses[(day, seq)][rel_frame_id, :, :]
                        temp_xyz = temp_pose[:3, 3]
                        dist = np.linalg.norm(temp_xyz - curr_xyz)
                        if dist < frames_interval:
                            continue
                        if dist > sequence_distance:
                            break

                        prev_frame_ids.append(rel_frame_id)
                        prev_img_paths.append(img_path)
                        lidar_path = os.path.join(self.root, day, seq, "velodyne_points", "data", f"{rel_frame_id:010d}.bin")
                        prev_lid_paths.append(lidar_path)
                        prev_poses.append(temp_pose)
                        prev_dists.append(dist)

                    # deal with next
                    cnt = 0
                    while True:
                        cnt += pos_step
                        rel_frame_id = frame_id + cnt

                        img_path = os.path.join(self.root, day, seq, image_folder, "data", f"{rel_frame_id:010d}.png")

                        if not os.path.exists(img_path):
                            break

                        temp_pose = self._poses[(day, seq)][rel_frame_id, :, :]
                        temp_xyz = temp_pose[:3, 3]
                        dist = np.linalg.norm(temp_xyz - curr_xyz)
                        if dist < frames_interval:
                            continue
                        if dist > sequence_distance:
                            break

                        next_frame_ids.append(rel_frame_id)
                        next_img_paths.append(img_path)
                        lidar_path = os.path.join(self.root, day, seq, "velodyne_points", "data", f"{rel_frame_id:010d}.bin")
                        next_lid_paths.append(lidar_path)
                        next_poses.append(temp_pose)
                        next_dists.append(dist)

                    prev_next_len = len(prev_poses) + len(next_poses)
                    
                    if prev_next_len > max_length:
                        max_length = prev_next_len
                    if prev_next_len < min_length:
                        min_length = prev_next_len
                    
                    stereo_sign = 'r' if is_right else 'l'
                    self.frame2scan.update({day + '/' + seq + "_" + str(frame_id) + '_' + stereo_sign: len(self.scans)})
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
                    self.scans.append({
                        "frame_id": frame_id,
                        "sequence": (day, seq),
                        "img_path": current_img_path,
                        "lid_path": current_lid_path,
                        "pose": curr_pose,
                        "is_right": is_right,

                        "prev_img_paths": prev_img_paths,
                        "prev_lid_paths": prev_lid_paths,
                        "next_img_paths": next_img_paths,
                        "next_lid_paths": next_lid_paths,

                        "T_velo_2_img": T_velo_2_img,
                        "T_velo_2_cam0": calib['P_v2c0'],
                        "T_velo_2_cam": T_cam0_2_cam2 @ calib['P_v2c0'],
                        "P": P,
                        "T_cam0_2_cam2": T_cam0_2_cam2,
                        "T_cam2_2_cam0": T_cam2_2_cam0,
                        'T_cam0_2_cam3': T_cam0_2_cam3,

                        "prev_poses": prev_poses,
                        "next_poses": next_poses,
                        "prev_dists": prev_dists,
                        "next_dists": next_dists,
                        "prev_frame_ids": prev_frame_ids,
                        "next_frame_ids": next_frame_ids
                    })
                
        print(min_length, max_length)

        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        data_path = Path(data_path)
        for day in data_path.iterdir():
            if not day.is_dir():
                continue
            day_sequences = [seq for seq in day.iterdir() if seq.is_dir()]
            lengths = [len(list((seq / "image_02" / "data").iterdir())) for seq in day_sequences]
            day_sequences = [(day.name, seq.name, length) for seq, length in zip(day_sequences, lengths)]
            all_sequences.extend(day_sequences)

        return all_sequences

    @staticmethod
    def _load_split(split_path):
        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            day, sequence = segments[0].split("/")
            # (day, sequence, id, is_right)
            return day, sequence, int(segments[1]), segments[2] == "r"

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path):
        calibs = {}

        for day in BASE_SIZES.keys():
            day_folder = Path(data_path) / day
            cam_calib_file = day_folder / "calib_cam_to_cam.txt"
            velo_calib_file = day_folder / "calib_velo_to_cam.txt"

            cam_calib_file_data = {}
            with open(cam_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        cam_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass
            velo_calib_file_data = {}
            with open(velo_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        velo_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass

            # Create 3x4 projection matrices
            P_rect_l = np.reshape(cam_calib_file_data['P_rect_02'], (3, 4))
            P_rect_l = np.vstack((P_rect_l, np.array([0, 0, 0, 1.0], dtype=np.float32)))
            P_rect_r = np.reshape(cam_calib_file_data['P_rect_03'], (3, 4))
            P_rect_r = np.vstack((P_rect_r, np.array([0, 0, 0, 1.0], dtype=np.float32)))

            R_rect = np.eye(4, dtype=np.float32)
            R_rect[:3, :3] = cam_calib_file_data['R_rect_00'].reshape(3, 3)

            T_v2c = np.hstack((velo_calib_file_data['R'].reshape(3, 3), velo_calib_file_data['T'][..., np.newaxis]))
            T_v2c = np.vstack((T_v2c, np.array([0, 0, 0, 1.0], dtype=np.float32)))

            P_v2c0 = R_rect @ T_v2c
            P_v2cl = P_rect_l @ P_v2c0
            P_v2cr = P_rect_r @ P_v2c0

            T_cam0_2_cam2 = np.eye(4)
            T_cam0_2_cam2[0, 3] = P_rect_l[0, 3] / P_rect_l[0, 0]
            T_cam0_2_cam3 = np.eye(4)
            T_cam0_2_cam3[0, 3] = P_rect_r[0, 3] / P_rect_r[0, 0]

            calibs[day] = {
                "P_v2c0": P_v2c0,
                "P_v2cl": P_v2cl,
                "P_v2cr": P_v2cr,
                "T_cam0_2_cam2": T_cam0_2_cam2,
                "T_cam0_2_cam3": T_cam0_2_cam3,
                "P_2": P_rect_l[:3, :3],
                "P_3": P_rect_r[:3, :3]}

        return calibs

    @staticmethod
    def _load_poses(pose_path, sequences):
        poses = {}

        for day, seq, _ in sequences:
            pose_file = Path(pose_path) / day / f"{seq}.txt"

            poses_seq = []
            try:
                with open(pose_file, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses_seq.append(T_w_cam0)

            except FileNotFoundError:
                pass
                # print(f'Ground truth poses are not avaialble for sequence {seq}.')

            poses_seq = np.array(poses_seq, dtype=np.float32)

            poses[(day, seq)] = poses_seq
        return poses

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
        mask = (depth < self.eval_depth) & (depth > 0.1)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        img_points[..., 0] = img_points[..., 0] / image_size[1]
        img_points[..., 1] = img_points[..., 1] / image_size[0]
        # img_points = np.round(img_points).astype(int)
        mask = mask & (img_points[..., 0] > 0) & \
                    (img_points[..., 1] > 0) & \
                    (img_points[..., 0] < 1) & \
                    (img_points[..., 1] < 1)
        
        if self.eigen_depth:
            mask = mask & (img_points[..., 0] > 0.03594771) & \
                        (img_points[..., 1] > 0.40810811) & \
                        (img_points[..., 0] < 0.96405229) & \
                        (img_points[..., 1] < 0.99189189)

        return img_points, depth, mask

    def load_depth(self, lidar_path, lidar2img, image_size):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        points = points[points[:, 0] >= 0, :]

        # project the points to the camera
        points = (self.transxy @ points.T).T # n, 4
        velo_pts_im = np.dot(lidar2img[0], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < image_size[1]) & (velo_pts_im[:, 1] < image_size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(image_size)
        depth[velo_pts_im[:, 1].astype(np.int64), velo_pts_im[:, 0].astype(np.int64)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (image_size[1] - 1) + velo_pts_im[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        if self.eigen_depth:
            mask = np.logical_and(depth > 1e-3, depth < self.eval_depth)
            crop = np.array([0.40810811 * image_size[0], 0.99189189 * image_size[0], 0.03594771 * image_size[1], 0.96405229 * image_size[1]]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            depth[~mask] = 0

        img_points = np.nonzero(depth)
        depth = depth[img_points][None, ...]
        img_points = np.stack((img_points[1], img_points[0]), axis=-1, dtype=np.float32)
        img_points[..., 0] = img_points[..., 0] / image_size[1]
        img_points[..., 1] = img_points[..., 1] / image_size[0]

        return img_points[None, ...], depth, np.ones_like(depth) > 0
    
    def __len__(self):
        return self.length
    
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
        lidar2img = scan['T_velo_2_img'] @ np.linalg.inv(self.transxy)
        img2lidar = np.linalg.inv(lidar2img)
                
        # img2prevImg = lidar2img @ \
        #     np.linalg.inv(anchor_scan['T_velo_2_cam0']) @ \
        #     np.linalg.inv(anchor_scan['prev_poses'][anchor_prev]) @ \
        #     anchor_scan['pose'] @ \
        #     anchor_scan['T_velo_2_cam0'] @ \
        #     img2lidar
        img2prevImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['prev_poses'][anchor_prev]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        # img2nextImg = lidar2img @ \
        #     np.linalg.inv(anchor_scan['T_velo_2_cam0']) @ \
        #     np.linalg.inv(anchor_scan['next_poses'][anchor_next]) @ \
        #     anchor_scan['pose'] @ \
        #     anchor_scan['T_velo_2_cam0'] @ \
        #     img2lidar
        img2nextImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['next_poses'][anchor_next]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        # temImg2lidar = np.linalg.inv(scan['T_velo_2_cam0']) @ \
        #     np.linalg.inv(scan['pose']) @ \
        #     anchor_scan['pose'] @ \
        #     anchor_scan['T_velo_2_cam0'] @ \
        #     img2lidar
        temImg2lidar = self.transxy @ np.linalg.inv(scan['T_velo_2_cam']) @ \
            scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(scan['pose']) @ \
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
            'temImg2lidar': [temImg2lidar],
            'intrinsic': intrinsic
        })

        return img_metas

    def read_surround_imgs(self, img_paths):
        imgs = []
        for filename in img_paths:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32))
        imgs = [img[:self.img_H, :self.img_W, :] for img in imgs]
        return imgs
    
    def to_scan_idx(self, index):
        day, seq, frame_id, is_right = self._datapoints[index]
        stereo_sign = 'r' if is_right else 'l'
        key = day + '/' + seq + "_" + str(frame_id) + '_' + stereo_sign
        return self.frame2scan[key]
    
    def append_self(self, flag, obj):
        obj[flag + '_img_paths'].append(obj['img_path'])
        obj[flag + '_lid_paths'].append(obj['lid_path'])
        obj[flag + '_poses'].append(obj['pose'])
        obj[flag + '_dists'].append(0.)
        obj[flag + '_frame_ids'].append(obj['frame_id'])
        return

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
            scan = deepcopy(self.scans[self.to_scan_idx(index)])
            scan_id = scan['frame_id']
            day, seq = scan['sequence']
            is_right = scan['is_right']
            stereo_sign = 'r' if is_right else 'l'

            if temporal_supervision == 'curr':
                anchor_scan = deepcopy(scan)
            elif temporal_supervision == 'prev':
                if len(scan['prev_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan = np.random.choice(scan['prev_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[day + '/' + seq + "_" + str(anchor_scan) + '_' + stereo_sign]])
            else:
                if len(scan['next_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan = np.random.choice(scan['next_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[day + '/' + seq + "_" + str(anchor_scan) + '_' + stereo_sign]])
            
            if len(anchor_scan['prev_frame_ids']) == 0 and len(anchor_scan['next_frame_ids']) == 0:
                index = np.random.randint(len(self))
                continue

            if len(anchor_scan['prev_frame_ids']) == 0:
                self.append_self('prev', anchor_scan)
                target_sign = 'ns'
            elif len(anchor_scan['next_frame_ids']) == 0:
                self.append_self('next', anchor_scan)
                target_sign = 'ps'
            else:
                if random.random() < 0.333:
                    target_sign = 'pn'
                elif random.random() < 0.5:
                    target_sign = 'ps'
                else:
                    target_sign = 'ns'

            if self.choose_nearest:
                anchor_prev = 0
                anchor_next = 0
            else:
                anchor_prev = np.random.randint(len(anchor_scan['prev_frame_ids']))
                anchor_next = np.random.randint(len(anchor_scan['next_frame_ids']))
            break
            
        img_metas = self.prepare_img_metas(
            scan, anchor_scan, anchor_prev, anchor_next)
        
        if target_sign == 'ns':
            anchor_img_path = img_metas['curr_imgs_path'][0]
            if stereo_sign == 'l':
                img_metas['prev_imgs_path'] = [anchor_img_path.replace('image_02', 'image_03')]
            else:
                img_metas['prev_imgs_path'] = [anchor_img_path.replace('image_03', 'image_02')]
            img_metas['img2prevImg'] = [
                img_metas['intrinsic'] @ \
                anchor_scan['T_cam0_2_cam3'] @ \
                anchor_scan['T_cam2_2_cam0'] @ \
                np.linalg.inv(img_metas['intrinsic'])]
        elif target_sign == 'ps':
            anchor_img_path = img_metas['curr_imgs_path'][0]
            if stereo_sign == 'l':
                img_metas['next_imgs_path'] = [anchor_img_path.replace('image_02', 'image_03')]
            else:
                img_metas['next_imgs_path'] = [anchor_img_path.replace('image_03', 'image_02')]
            img_metas['img2nextImg'] = [
                img_metas['intrinsic'] @ \
                anchor_scan['T_cam0_2_cam3'] @ \
                anchor_scan['T_cam2_2_cam0'] @ \
                np.linalg.inv(img_metas['intrinsic'])]
        
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                scan['lid_path'], img_metas['lidar2img'], [self.img_H, self.img_W])
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})

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
    data_path = 'data/kitti_raw/'
    img_size = [352, 1216]

    dataset = Kitti_Raw(
        root = data_path,
        pose_path = 'dataset/kitti_raw/orb-slam_poses',
        split_path = 'dataset/kitti_raw/splits/eigen_zhou/train_files.txt',
        frames_interval=0.4,
        sequence_distance=10,
        cur_prob=0.333,
        crop_size=img_size,
        strict=True,
        return_depth=False,
        prev_prob=0.5,
        choose_nearest = True,
    )
    
    batch = dataset[0]
    
    import pdb; pdb.set_trace()