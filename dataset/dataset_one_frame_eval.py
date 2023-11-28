import os, numpy as np, pickle, random
from mmcv.image.io import imread
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET

if 'HFAI' in os.environ:
    hfai = True
    from dataset.loading import LoadMultiViewImageFromFilesHF, \
        LoadPtsFromFilesHF
else:
    hfai = False


@OPENOCC_DATASET.register_module()
class nuScenes_One_Frame_Eval:
    def __init__(
            self, 
            data_path, 
            imageset='train', 
            nusc=None,
            crop_size=[768, 1600],
            cam_types=None,
            eval_depth=80,
            scene_name=None,
            **kwargs):

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        if scene_name is not None:
            assert isinstance(scene_name, str) and scene_name.startswith('scene-')
            selected_indices = data['scene_info'][scene_name]
            self.selected_indices = list(range(selected_indices[0], selected_indices[1] + 1))
        else:
            self.selected_indices = list(range(len(self.nusc_infos)))

        self.data_path = data_path
        self.nusc = nusc
        self.crop_size = crop_size

        self.cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ] if cam_types is None else cam_types
        self.eval_depth = eval_depth

        if hfai:
            self.img_loader = LoadMultiViewImageFromFilesHF(
                to_float32=True,
                file_client_args=dict(
                    backend='ffrecord',
                    fname=data_path+'CAM',
                    filename2idx='img_fname2idx.pkl'))
            self.pts_loader = LoadPtsFromFilesHF(
                to_float32=True,
                file_client_args=dict(
                    backend='ffrecord',
                    fname=data_path+'LIDAR',
                    filename2idx='pts_fname2idx.pkl'))
        else:
            self.img_loader = self.pts_loader = None
        
        self.lidar2cam_rect = np.eye(4)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.selected_indices)
    
    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size):
        # lidar2img: N, 4, 4
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(scan.T, 0) # 1, 4, n
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

    def __getitem__(self, index):
        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        while True:
            index = self.selected_indices[index]
            info = deepcopy(self.nusc_infos[index])
            if len(info['nice_neighbor_prev']) == 0 and len(info['nice_neighbor_next']) ==0:
                index = np.random.randint(len(self))
                continue
            break

        #### 3. prepare img_metas
        imgs_info = self.get_data_info(info)
        img_metas = {
            'input_imgs_path': imgs_info['img_filename'],
            'lidar2img': imgs_info['lidar2img'],
            'img2lidar': imgs_info['img2lidar'],
            'ego2lidar': imgs_info['ego2lidar'],
            'token': info['token'],
            'timestamp': info['timestamp'],
            'intrinsic': imgs_info['cam_intrinsic']}

        anchor_imgs = []                       # list[list[array]]
        anchor_depth_locs = []                 # list[array]
        anchor_depth_gts = []                  # list[array]
        anchor_depth_masks = []                # list[array]
        temImg2lidars = []                     # list[list[array]]
        anchor_frame_dist = \
            info['prev_dists'] + info['next_dists'] # list[float]
        
        for anchor in info['nice_neighbor_prev'] + info['nice_neighbor_next']:
            anchor_info = deepcopy(self.nusc_infos[anchor])
            anchor_imgs_info = self.get_data_info_temporal(info, anchor_info)
            anchor_img_path = anchor_imgs_info['image_paths']
            temImg2lidar = anchor_imgs_info['temImg2lidar']
            anchor_img = self.read_surround_imgs(anchor_img_path)
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                anchor_info['lidar_path'], img_metas['lidar2img'], self.crop_size)
                        
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
        
        #### 4. read imgs
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'])
        data_tuple = (input_imgs, anchor_imgs, img_metas)
        return data_tuple   
    
    def read_surround_imgs(self, img_paths):
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32))
        imgs = [img[:self.crop_size[0], :self.crop_size[1], :] for img in imgs]
        return imgs

    def get_data_info_temporal(self, info, info_tem):
        image_paths = []
        temImg2lidars = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

        ego2global_r = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global = np.eye(4)
        ego2global[:3, :3] = ego2global_r
        ego2global[:3, 3] = np.array(info['ego2global_translation']).T

        lidar2global = ego2global @ lidar2ego
        global2lidar = np.linalg.inv(lidar2global)
        
        for cam_type_tem, cam_info_tem in info_tem['cams'].items():
            image_paths.append(cam_info_tem['data_path'])

            temIntrinsic = cam_info_tem['cam_intrinsic']
            temImg2temSensor = np.eye(4)
            temImg2temSensor[:3, :3] = temIntrinsic
            temImg2temSensor = np.linalg.inv(temImg2temSensor)

            temSensor2temEgo_r = Quaternion(cam_info_tem['sensor2ego_rotation']).rotation_matrix
            temSensor2temEgo = np.eye(4)
            temSensor2temEgo[:3, :3] = temSensor2temEgo_r
            temSensor2temEgo[:3, 3] = np.array(cam_info_tem['sensor2ego_translation']).T

            temEgo2global_r = Quaternion(cam_info_tem['ego2global_rotation']).rotation_matrix
            temEgo2global = np.eye(4)
            temEgo2global[:3, :3] = temEgo2global_r
            temEgo2global[:3, 3] = np.array(cam_info_tem['ego2global_translation']).T

            temImg2global = temEgo2global @ temSensor2temEgo @ temImg2temSensor

            temImg2lidar = global2lidar @ temImg2global
            temImg2lidars.append(temImg2lidar)

        out_dict = dict(
            image_paths=image_paths,
            temImg2lidar=np.asarray(temImg2lidars)
        )
        return out_dict

    def get_data_info(self, info):

        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])

            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            img2lidar_rt = np.linalg.inv(lidar2img_rt)

            cam_intrinsics.append(viewpad)

            lidar2img_rts.append(lidar2img_rt)
            img2lidar_rts.append(img2lidar_rt)

        input_dict =dict(
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ego2lidar=ego2lidar)
        return input_dict
    
