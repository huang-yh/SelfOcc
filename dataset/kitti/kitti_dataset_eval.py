import glob
import os
import time

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
class Kitti_Novel_View_Eval:
    def __init__(
            self,
            split,
            root, 
            preprocess_root,
            frames_interval=0.4,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            sequences=None,
            selected_frames=None, 
            crop_size=[370, 1220],
            **kwargs,
    ):
        self.root = root
        self.preprocess_root = preprocess_root
        self.depth_preprocess_root = os.path.join(preprocess_root, "depth")
        self.transform_preprocess_root = os.path.join(preprocess_root, "transform")
        self.n_classes = 20
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        
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
        self.sequence_distance = sequence_distance

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
            
            dists_glob_path = os.path.join(
                self.root, "dataset", "sequences", sequence, "image_2", "*.png"
            )
            
            seq_img_paths = glob.glob(glob_path)
            dists_seq_img_paths = glob.glob(dists_glob_path)

            max_length = 0
            min_length = 50
            paired_dists = {}
            # dist_step = 1 if split == 'train' else 5
            dist_step = 1
            for seq_img_path in dists_seq_img_paths:
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
                prev_poses, next_poses = [gt_global_poses[int(frame_id)]], [gt_global_poses[int(frame_id)]]
                prev_dists, next_dists = [], []

                # if self.split == 'train':
                #     pos_step = 1
                #     neg_step = -1
                # else:
                #     pos_step = 5
                #     neg_step = -5
                pos_step = 1
                neg_step = -1
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

                    prev_pose = prev_poses[-1]
                    prev_xyz = dump_xyz(prev_pose)
                    current_xyz = dump_xyz(gt_global_poses[int(rel_frame_id)])
                    tmp_dist = np.sqrt(
                        (prev_xyz[0] - current_xyz[0]) ** 2 + (prev_xyz[2] - current_xyz[2]) ** 2)
                    # tmp_dist = paired_dists["{:06d}".format(int(rel_frame_id) + pos_step)]
                    dist += tmp_dist
                    # if dist < frames_interval:
                    if tmp_dist < frames_interval:
                        continue
                    if dist > sequence_distance:
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

                    prev_pose = next_poses[-1]
                    prev_xyz = dump_xyz(prev_pose)
                    current_xyz = dump_xyz(gt_global_poses[int(rel_frame_id)])
                    tmp_dist = np.sqrt(
                        (prev_xyz[0] - current_xyz[0]) ** 2 + (prev_xyz[2] - current_xyz[2]) ** 2)
                    # tmp_dist = paired_dists[rel_frame_id]
                    dist += tmp_dist
                    # if dist < frames_interval:
                    if tmp_dist < frames_interval:
                        continue
                    if dist > sequence_distance:
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

                is_included = len(next_frame_ids) > 0 if selected_frames is None else frame_id in selected_frames

                if is_included:

                    # prev_next_len = len(prev_poses) + len(next_poses)
                    prev_next_len = len(next_frame_ids)
                    
                    if prev_next_len > max_length:
                        max_length = prev_next_len
                    if prev_next_len < min_length:
                        min_length = prev_next_len

                    self.frame2scan.update({str(sequence) + "_" + frame_id: len(self.scans)})
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

                            "prev_poses": prev_poses[1:],
                            "next_poses": next_poses[1:],
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
    
    def __len__(self):
        return len(self.scans)
    
    def prepare_temImg2lidar(self, scan, anchor):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = scan['P'][:3, :3]
                
        temImg2lidar = self.transxy @ np.linalg.inv(scan['T_velo_2_cam']) @ \
            scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(scan['pose']) @ \
            scan['next_poses'][anchor] @ \
            np.linalg.inv(scan['T_cam0_2_cam2']) @ \
            np.linalg.inv(intrinsic)
        
        return [temImg2lidar]

    def prepare_img_metas(self, scan):
        img_metas = {}
        img_metas.update({
            'input_imgs_path': [scan['img_path']]})

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = scan['P'][:3, :3]
        lidar2img = intrinsic @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        img2lidar = np.linalg.inv(lidar2img)
                
        img_metas.update({
            'lidar2img': np.expand_dims(lidar2img, axis=0),
            'img2lidar': [img2lidar],
            'token': scan['frame_id'],
            'sequence': scan['sequence']
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

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        scan = deepcopy(self.scans[index])
            
        img_metas = self.prepare_img_metas(scan)

        anchor_imgs = []                       # list[list[array]]
        anchor_depth_locs = []                 # list[array]
        anchor_depth_gts = []                  # list[array]
        anchor_depth_masks = []                # list[array]
        temImg2lidars = []                     # list[list[array]]
        anchor_frame_dist = scan['next_dists'] # list[float]

        for anchor in range(len(scan['next_frame_ids'])):
            anchor_img_path = scan['next_img_paths'][anchor]
            anchor_lid_path = scan['next_lid_paths'][anchor]
            temImg2lidar = self.prepare_temImg2lidar(scan, anchor)
            anchor_img = self.read_surround_imgs([anchor_img_path])
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                anchor_lid_path, img_metas['lidar2img'], [self.img_H, self.img_W])
            
            anchor_imgs.append(anchor_img)
            anchor_depth_locs.append(depth_loc)
            anchor_depth_gts.append(depth_gt)
            anchor_depth_masks.append(depth_mask)
            temImg2lidars.append(temImg2lidar)

        img_metas.update({
            'depth_loc': anchor_depth_locs,
            'depth_gt': anchor_depth_gts,
            'depth_mask': anchor_depth_masks,
            'temImg2lidars': temImg2lidars,
            'frame_dists': anchor_frame_dist
        })

        # read 6 cams
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'])

        data_tuple = (input_imgs, anchor_imgs, img_metas)
        return data_tuple   

        # frame_id = scan['frame_id']
        # sequence = scan['sequence']
        # lidar_paths = scan['lidar_paths']
        # rel_frame_ids = scan['rel_frame_ids']
        # distances = scan['distances']
        # infer_id = 0
        

        # P = scan["P"]
        # T_velo_2_cam = scan["T_velo_2_cam"]

        # img_paths = scan["img_paths"]

        # img_sources = []
        # img_input_sources = []
        # img_targets = []
        # lidar_depths = []
        # depths = []
        # loc2d_with_depths = []
        # T_source2infers = []
        # T_source2targets = []
        # source_distances = []
        # source_frame_ids = []

        
        # n_sources = min(len(distances) - 1, self.n_sources)
        
        # for d_id in range(n_sources):
        #     if self.n_sources < len(distances):    
        #         source_id = np.random.randint(1, len(distances))
        #         source_distance = distances[source_id]  
        #     else:
        #         source_id = d_id + 1
        #         source_distance = distances[source_id]
        
        #     source_distances.append(source_distance)

        #     rel_frame_id = rel_frame_ids[source_id]
        #     source_frame_ids.append(rel_frame_id)

        #     target_id = source_id - 1

        #     img_input_source = self.to_tensor_normalized(read_rgb(img_paths[source_id]))
        #     img_input_sources.append(img_input_source)

       
        #     img_source = self.to_tensor(read_rgb(img_paths[source_id]))
        #     img_target = self.to_tensor(read_rgb(img_paths[target_id]))


        #     lidar_path = lidar_paths[source_id]
        #     loc2d_with_depth, lidar_depth, _ = self.get_depth_from_lidar(lidar_path, P, T_velo_2_cam,
        #                                                                  (self.img_W, self.img_H))

        #     if self.n_rays  < lidar_depth.shape[0]:
        #         idx = np.random.choice(lidar_depth.shape[0], size=self.n_rays, replace=False)
        #         loc2d_with_depth = loc2d_with_depth[idx, :]
        #         lidar_depth = lidar_depth[idx]

        #     img_sources.append(img_source)
        #     img_targets.append(img_target)
        #     lidar_depths.append(torch.from_numpy(lidar_depth))
        #     loc2d_with_depths.append(torch.from_numpy(loc2d_with_depth))

        #     # Get transformation from source to target coord
        #     transform_dir = os.path.join(self.transform_preprocess_root,
        #                                  "{}_{}_all".format(sequence, self.frames_interval))
        #     os.makedirs(transform_dir, exist_ok=True)

        #     transform_path = os.path.join(transform_dir, "{}.pkl".format(frame_id))

            
        #     if os.path.exists(transform_path):
        #         try:
        #             with open(transform_path, "rb") as input_file:
        #                 transform_data = pickle.load(input_file)
        #         except EOFError:
        #             transform_data = {}
        #     else:
        #         transform_data = {}

        #     if '{}'.format(source_id) in transform_data:
        #         T_out_dict = transform_data['{}'.format(source_id)]
        #     else:
        #         poses = scan["poses"]
        #         pose_source = poses[source_id]
        #         pose_infer = poses[infer_id]
        #         pose_target = poses[target_id]
        #         lidar_path_source = lidar_paths[source_id]
        #         lidar_path_target = lidar_paths[target_id]
        #         lidar_path_infer = lidar_paths[infer_id]

        #         T_out_dict = compute_transformation(
        #             lidar_path_source, lidar_path_infer, lidar_path_target,
        #             pose_source, pose_infer, pose_target,
        #             T_velo_2_cam, scan['T_cam0_2_cam2'])

        #         transform_data['{}'.format(source_id)] = T_out_dict
        #         with open(transform_path, "wb") as input_file:
        #             pickle.dump(transform_data, input_file)
        #             print("{}: saved {} to {}".format(frame_id, source_id, transform_path))

        #     T_source2infer = T_out_dict['T_source2infer']
        #     T_source2target = T_out_dict['T_source2target']
        #     T_source2infers.append(torch.from_numpy(T_source2infer).float())
        #     T_source2targets.append(torch.from_numpy(T_source2target).float())
           


        # data = {

        #     "img_input_sources": img_input_sources,
        #     "source_distances": source_distances,
        #     "source_frame_ids": source_frame_ids,

        #     "img_sources": img_sources,
        #     "img_targets": img_targets,

        #     "lidar_depths": lidar_depths,
        #     "depths": depths,
        #     "loc2d_with_depths": loc2d_with_depths,
        #     "T_source2infers": T_source2infers,
        #     "T_source2targets": T_source2targets,

        #     "frame_id": frame_id,
        #     "sequence": sequence,

        #     "P": P,
        #     "T_velo_2_cam": T_velo_2_cam,
        #     "T_cam2_2_cam0": scan['T_cam2_2_cam0'],
        #     "T_cam0_2_cam2": scan['T_cam0_2_cam2'],

        # }
        
        # scale_3ds = [self.output_scale]
        # data["scale_3ds"] = scale_3ds
        # cam_K = P[0:3, 0:3]
        # data["cam_K"] = cam_K
        # for scale_3d in scale_3ds:
        #     # compute the 3D-2D mapping
           
        #     projected_pix, fov_mask, sensor_distance = vox2pix(
        #         T_velo_2_cam,
        #         cam_K,
        #         self.vox_origin,
        #         self.voxel_size * scale_3d,
        #         self.img_W,
        #         self.img_H,
        #         self.scene_size,
        #     )
           
        #     data["projected_pix_{}".format(scale_3d)] = projected_pix
        #     data["sensor_distance_{}".format(scale_3d)] = sensor_distance
        #     data["fov_mask_{}".format(scale_3d)] = fov_mask
        
        # img_input = read_rgb(img_paths[infer_id])

        # img_input = self.to_tensor_normalized(img_input)
        # data["img_input"] = img_input
        

        # label_path = os.path.join(
        #     self.root, "dataset", "sequences", sequence, "voxels", "{}.label".format(frame_id)
        # )
        # invalid_path = os.path.join(
        #     self.root, "dataset", "sequences", sequence, "voxels", "{}.invalid".format(frame_id)
        # )
        # # data['target_1_1'] = self.read_semKITTI_label(label_path, invalid_path)
        
        
        # return data


    # @staticmethod
    # def read_semKITTI_label(label_path, invalid_path):
    #     remap_lut = SemanticKittiIO.get_remap_lut("./scenerf/data/semantic_kitti/semantic-kitti.yaml")
    #     LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
    #     INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
    #     LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
    #         np.float32
    #     )  # Remap 20 classes semanticKITTI SSC
       
    #     LABEL[
    #         np.isclose(INVALID, 1)
    #     ] = 255  # Setting to unknown all voxels marked on invalid mask...
        
    #     LABEL = LABEL.reshape(256,256, 32)
    #     return LABEL

    # def __len__(self):
    #     return len(self.scans)

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