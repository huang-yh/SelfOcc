import os, numpy as np, random, mmengine, math
from mmcv.image.io import imread
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET


def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


@OPENOCC_DATASET.register_module()
class nuScenes_One_Frame_Sweeps_Dist_Vis:
    def __init__(
            self, 
            data_path, 
            imageset='train', 
            crop_size=[768, 1600],
            return_depth=False,
            eval_depth=80,
            ego_centric=False,
            scene_token=None,
            **kwargs):
        
        assert scene_token is not None

        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        assert scene_token in self.scene_infos
        self.infos = self.scene_infos[scene_token]

        self.data_path = data_path
        self.crop_size = crop_size
        self.return_depth = return_depth
        self.eval_depth = eval_depth
        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.ego_centric = ego_centric
        self.img_loader = self.pts_loader = None
        
        self.lidar2cam_rect = np.eye(4)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.infos)
    
    def __getitem__(self, index):
        info = deepcopy(self.infos[index])
        imgs_info = self.get_data_info(info)

        img_metas = {
            'input_imgs_path': imgs_info['img_filename'],
            'lidar2img': imgs_info['lidar2img'],
            'img2lidar': imgs_info['img2lidar'],
            'intrinsic': imgs_info['cam_intrinsic'],
            'cam2ego': imgs_info['cam2ego'],
            'ego2lidar': imgs_info['ego2lidar'],
            # 'token': info['token'],
            'timestamp': info['timestamp'],
            'cam_positions': imgs_info['cam_positions'],
            'focal_positions': imgs_info['focal_positions']
            }
                    
        if self.ego_centric:
            ego2lidar = img_metas['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            ego2img = img_metas['lidar2img'] @ ego2lidar[None, ...]
            img2ego = lidar2ego[None, ...] @ img_metas['img2lidar']
            img_metas.update({
                'lidar2img': ego2img,
                'img2lidar': img2ego})
        
        #### 4. read imgs
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'])
        
        data_tuple = (input_imgs, img_metas)
        return data_tuple   
    
    def read_surround_imgs(self, img_paths):
        imgs = []
        for filename in img_paths:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32))
        imgs = [img[:self.crop_size[0], :self.crop_size[1], :] for img in imgs]
        return imgs

    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            cam2ego_r = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = cam2ego_r
            cam2ego[:3, 3] = np.array(info['data'][cam_type]['calib']['translation']).T

            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic

            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

            lidar2img_rts.append(lidar2img)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            
        input_dict =dict(
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts),
            cam_positions=cam_positions,
            focal_positions=focal_positions)
        return input_dict
