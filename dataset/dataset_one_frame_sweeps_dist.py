import os, numpy as np, random, mmengine, math
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
class nuScenes_One_Frame_Sweeps_Dist:
    def __init__(
            self, 
            data_path, 
            imageset='train', 
            crop_size=[768, 1600],
            input_img_crop_size=None,
            min_dist=0.4,
            max_dist=10.0,

            strict=True,
            return_depth=False,
            eval_depth=80,
            cur_prob=1.0,
            prev_prob=0.5,
            choose_nearest=False,
            ref_sensor='CAM_FRONT',
            composite_prev_next=False,
            sensor_mus=[3.0, 0.5],
            sensor_sigma=0.5,
            ego_centric=False,
            **kwargs):

        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']

        self.data_path = data_path
        self.crop_size = crop_size
        self.input_img_crop_size = crop_size if input_img_crop_size is None else input_img_crop_size
        self.strict = strict
        self.return_depth = return_depth
        self.eval_depth = eval_depth
        self.cur_prob = cur_prob
        self.prev_prob = prev_prob
        self.choose_nearest = choose_nearest
        self.composite_prev_next = composite_prev_next
        self.sensor_mus = {
            'CAM_FRONT': sensor_mus[0], 'CAM_FRONT_RIGHT': sensor_mus[1], 'CAM_FRONT_LEFT': sensor_mus[1],
            'CAM_BACK': sensor_mus[0], 'CAM_BACK_LEFT': sensor_mus[1], 'CAM_BACK_RIGHT': sensor_mus[1]
        }
        self.sensor_sigma = sensor_sigma
        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.ego_centric = ego_centric

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

        #### collect temporal information
        for scene_idx, (scene_token, scene_samples) in enumerate(self.scene_infos.items()):
            if (scene_idx + 1) % 50 == 0:
                print(f'One_Frame_Sweeps: processed {scene_idx + 1} scenes.')
            length = len(scene_samples)
            for i, sample in enumerate(scene_samples):
                curr_xyz = get_xyz(sample['data'][ref_sensor]['pose'])
                prev_samples, next_samples = [], []
                prev_dists, next_dists = [], []
                for j in range(i - 1, -1, -1):
                    temp_xyz = get_xyz(scene_samples[j]['data'][ref_sensor]['pose'])
                    temp_dist = np.linalg.norm(curr_xyz - temp_xyz)
                    if temp_dist > max_dist:
                        break
                    if temp_dist > min_dist:
                        prev_samples.append((scene_token, j))
                        prev_dists.append(temp_dist)
                
                for j in range(i + 1, length, 1):
                    temp_xyz = get_xyz(scene_samples[j]['data'][ref_sensor]['pose'])
                    temp_dist = np.linalg.norm(curr_xyz - temp_xyz)
                    if temp_dist > max_dist:
                        break
                    if temp_dist > min_dist:
                        next_samples.append((scene_token, j))
                        next_dists.append(temp_dist)
                
                if not strict:
                    prev_samples.append((scene_token, i))
                    prev_dists.append(0.)
                    next_samples.append((scene_token, i))
                    next_dists.append(0.)
                
                sample.update({
                    'prev_samples': prev_samples,
                    'prev_dists': prev_dists,
                    'next_samples': next_samples,
                    'next_dists': next_dists,})

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.keyframes)
    
    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size):
        # lidar2img: N, 4, 4
        scan = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(scan.T, 0) # 1, 4, n
        img_points = np.transpose(lidar2img @ points_hcoords, (0, 2, 1)) # N, n, 4

        depth = img_points[..., 2] # N, n
        img_points = img_points[..., :2] # N, n, 2
        # mask = (depth < self.eval_depth) & (depth > 1e-3)  # get points with depth < max_sample_depth
        mask = (depth < self.eval_depth) & (depth > 1.0)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        img_points[..., 0] = img_points[..., 0] / image_size[1]
        img_points[..., 1] = img_points[..., 1] / image_size[0]
        # img_points = np.round(img_points).astype(int)
        mask = mask & (img_points[..., 0] > 0) & \
                    (img_points[..., 1] > 0) & \
                    (img_points[..., 0] < 1) & \
                    (img_points[..., 1] < 1)

        return img_points, depth, mask
    
    def composite_dict(self, anchor_info):
        datas = []
        for prefix in ['prev_', 'next_']:
            data = dict()
            dists = np.asarray(anchor_info[prefix + 'dists'])
            for sensor_type in self.sensor_types:
                mu = self.sensor_mus[sensor_type]
                sigma = self.sensor_sigma
                probs = 1 / math.sqrt(2 * math.pi) / sigma * np.exp(-1 / (2 * sigma * sigma) * ((dists - mu) ** 2))
                probs = probs / np.sum(probs)
                idx = np.random.choice(len(dists), p=probs)
                scene_token, sample_idx = anchor_info[prefix + 'samples'][idx]
                data.update({sensor_type: self.scene_infos[scene_token][sample_idx]['data'][sensor_type]})
            datas.append(data)
        return {'data': datas[0]}, {'data': datas[1]}

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
            scene_token, index = self.keyframes[index]
            info = deepcopy(self.scene_infos[scene_token][index])

            if temporal_supervision == 'curr':
                anchor_info = deepcopy(info)
            elif temporal_supervision == 'prev':
                if len(info['prev_samples']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scene_token, anchor_info_id = info['prev_samples'][np.random.randint(len(info['prev_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['prev_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id <= index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])
            else:
                if len(info['next_samples']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scene_token, anchor_info_id = info['next_samples'][np.random.randint(len(info['next_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['next_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id >= index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])

            if len(anchor_info['prev_samples']) == 0 or \
                len(anchor_info['next_samples']) == 0:
                index = np.random.randint(len(self))
                continue

            if self.composite_prev_next:
                anchor_prev, anchor_next = self.composite_dict(anchor_info)
            else:
                if self.choose_nearest:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][0]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][0]
                else:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][np.random.randint(len(anchor_info['prev_samples']))]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][np.random.randint(len(anchor_info['next_samples']))]
                    # anchor_prev_scene_token, anchor_prev_idx = np.random.choice(anchor_info['prev_samples'])
                    # anchor_next_scene_token, anchor_next_idx = np.random.choice(anchor_info['next_samples'])
                assert anchor_prev_scene_token == scene_token and \
                    anchor_next_scene_token == scene_token
                anchor_prev = deepcopy(self.scene_infos[scene_token][anchor_prev_idx])
                anchor_next = deepcopy(self.scene_infos[scene_token][anchor_next_idx])
            break

        #### 3. prepare img_metas
        imgs_info = self.get_data_info(info)
        anchor_dict = self.get_data_info_anchor(info, anchor_info)
        prev_dict = self.get_data_info_temporal(anchor_info, anchor_prev)
        next_dict = self.get_data_info_temporal(anchor_info, anchor_next)

        img_metas = {
            'input_imgs_path': imgs_info['img_filename'],
            'curr_imgs_path': anchor_dict['image_paths'],
            'prev_imgs_path': prev_dict['image_paths'],
            'next_imgs_path': next_dict['image_paths'],
            'lidar2img': imgs_info['lidar2img'],
            'img2lidar': imgs_info['img2lidar'],
            'intrinsic': imgs_info['cam_intrinsic'],
            'cam2ego': imgs_info['cam2ego'],
            'temImg2lidar': anchor_dict['temImg2lidar'],
            'ego2lidar': imgs_info['ego2lidar'],
            'token': info['token'],
            'timestamp': info['timestamp'],
            'img2prevImg': prev_dict['img2temImg'],
            'img2nextImg': next_dict['img2temImg'],}
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                info['data']['LIDAR_TOP']['filename'], img_metas['lidar2img'], self.crop_size)
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})
            
        if self.ego_centric:
            ego2lidar = img_metas['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            ego2img = img_metas['lidar2img'] @ ego2lidar[None, ...]
            img2ego = lidar2ego[None, ...] @ img_metas['img2lidar']
            temImg2ego = lidar2ego[None, ...] @ img_metas['temImg2lidar']
            img_metas.update({
                'lidar2img': ego2img,
                'img2lidar': img2ego,
                'temImg2lidar': temImg2ego,
                'ego2lidar': np.eye(4)})
        
        #### 4. read imgs
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'], self.input_img_crop_size)
        curr_imgs = self.read_surround_imgs(img_metas['curr_imgs_path'], self.crop_size)
        prev_imgs = self.read_surround_imgs(img_metas['prev_imgs_path'], self.crop_size)
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'], self.crop_size)
        
        data_tuple = (
            [input_imgs, curr_imgs, prev_imgs, next_imgs], img_metas)
        return data_tuple   
    
    def read_surround_imgs(self, img_paths, crop_size):
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32))
        imgs = [img[:crop_size[0], :crop_size[1], :] for img in imgs]
        return imgs

    def get_data_info_temporal(self, info, info_tem):
        image_paths = []
        img2temImgs = []

        for cam_type in self.sensor_types:

            cam_info_tem = info_tem['data'][cam_type]
            cam_info = info['data'][cam_type]
            image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

            temImg2global = get_img2global(cam_info_tem['calib'], cam_info_tem['pose'])
            img2global = get_img2global(cam_info['calib'], cam_info['pose'])

            img2temImg = np.linalg.inv(temImg2global) @ img2global            
            img2temImgs.append(img2temImg)

        out_dict = dict(
            image_paths=image_paths,
            img2temImg=np.asarray(img2temImgs))
        return out_dict
    
    def get_data_info_anchor(self, info, info_tem):
        image_paths = []
        temImg2lidars = []

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])

        for cam_type in self.sensor_types:

            cam_info_tem = info_tem['data'][cam_type]
            image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

            temImg2global = get_img2global(cam_info_tem['calib'], cam_info_tem['pose'])

            temImg2lidar = np.linalg.inv(lidar2global) @ temImg2global
            temImg2lidars.append(temImg2lidar)

        out_dict = dict(
            image_paths=image_paths,
            temImg2lidar=np.asarray(temImg2lidars))
        return out_dict

    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []

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
            cam2ego=np.asarray(cam2ego_rts))
        return input_dict

    # def get_data_info(self, info):
    #     image_paths = []
    #     lidar2img_rts = []
    #     img2lidar_rts = []
    #     cam_intrinsics = []
    #     cam2ego_rts = []

    #     lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
    #     lidar2ego = np.eye(4)
    #     lidar2ego[:3, :3] = lidar2ego_r
    #     lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
    #     ego2lidar = np.linalg.inv(lidar2ego)

    #     for cam_type in self.sensor_types:
    #         image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

    #         cam2ego_r = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
    #         cam2ego = np.eye(4)
    #         cam2ego[:3, :3] = cam2ego_r
    #         cam2ego[:3, 3] = np.array(info['data'][cam_type]['calib']['translation']).T
    #         ego2cam = np.linalg.inv(cam2ego)
    #         lidar2cam = ego2cam @ lidar2ego

    #         intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
    #         viewpad = np.eye(4)
    #         viewpad[:3, :3] = intrinsic
    #         lidar2img = viewpad @ lidar2cam
    #         img2lidar = np.linalg.inv(lidar2img)

    #         lidar2img_rts.append(lidar2img)
    #         img2lidar_rts.append(img2lidar)
    #         cam_intrinsics.append(viewpad)
    #         cam2ego_rts.append(cam2ego)
            
    #     input_dict =dict(
    #         img_filename=image_paths,
    #         lidar2img=np.asarray(lidar2img_rts),
    #         img2lidar=np.asarray(img2lidar_rts),
    #         cam_intrinsic=np.asarray(cam_intrinsics),
    #         ego2lidar=ego2lidar,
    #         cam2ego=np.asarray(cam2ego_rts))
    #     return input_dict
    

if __name__ == '__main__':
    import torch
    dataset = nuScenes_One_Frame_Sweeps(
        'data/nuscenes',
        'data/nuscenes_infos_val_temporal.pkl',
    )

    batch = dataset[300]
    imgs, img_metas, points = batch
    curr_imgs, prev_imgs, next_imgs = imgs

    def list2tensor(imgs):
        imgs = np.asarray(imgs)
        imgs = torch.from_numpy(imgs)
        return imgs
    curr_imgs = list2tensor(curr_imgs)
    prev_imgs = list2tensor(prev_imgs)
    next_imgs = list2tensor(next_imgs)
    
    print('img metas contains keys: ', list(img_metas.keys()))
    l2i = list2tensor(img_metas['lidar2img'])
    img2prevImg = list2tensor(img_metas['img2prevImg'])
    img2nextImg = list2tensor(img_metas['img2nextImg'])
    
    def get_diagonal(trans):
        trans = trans.reshape(6, 6, 4, 4)
        trans = torch.diagonal(trans, 0, 0, 1)
        trans = trans.permute(2, 0, 1)
        return trans
    img2prevImg = get_diagonal(img2prevImg)
    img2nextImg = get_diagonal(img2nextImg)

    def filter_invalid_pts(pts):
        mask = torch.norm(pts, dim=-1)
        mask = mask > 2
        pts = pts[mask]
        return pts
    points = torch.from_numpy(points)
    points = filter_invalid_pts(points)
    points = torch.cat([
        points,
        torch.ones_like(points[:, :1])
    ], dim=1)

    curr_pts_uv = torch.matmul(
        l2i.unsqueeze(1).to(points.dtype), 
        points.reshape(1, -1, 4, 1))
    
    prev_pts_uv = torch.matmul(
        img2prevImg.unsqueeze(1).to(points.dtype),
        curr_pts_uv)
    
    next_pts_uv = torch.matmul(
        img2nextImg.unsqueeze(1).to(points.dtype),
        curr_pts_uv)
    
    curr_pts_uv = curr_pts_uv.squeeze(-1)
    prev_pts_uv = prev_pts_uv.squeeze(-1)
    next_pts_uv = next_pts_uv.squeeze(-1)

    def get_pixel(uv, eps=1e-5):
        uv = uv[..., :2] / torch.maximum(
            uv[..., 2:3],
            torch.ones_like(uv[..., :1] * eps))
        return uv
    
    curr_pts_uv = get_pixel(curr_pts_uv)
    prev_pts_uv = get_pixel(prev_pts_uv)
    next_pts_uv = get_pixel(next_pts_uv)

    mask1 = torch.logical_and(
        curr_pts_uv[0, :, 0] > 600,
        curr_pts_uv[0, :, 0] < 1000)
    mask2 = torch.logical_and(
        curr_pts_uv[0, :, 1] > 400,
        curr_pts_uv[0, :, 1] < 500)
    mask = torch.logical_and(mask1, mask2)
    curr_pts_uv = curr_pts_uv[:, mask, :]
    prev_pts_uv = prev_pts_uv[:, mask, :]
    next_pts_uv = next_pts_uv[:, mask, :]

    def filter_invalid_points(uv):
        uv[..., 0] = torch.clamp(uv[..., 0], 0, 1600)
        uv[..., 1] = torch.clamp(uv[..., 1], 0, 900)
        return uv
    curr_pts_uv = filter_invalid_points(curr_pts_uv)
    prev_pts_uv = filter_invalid_points(prev_pts_uv)
    next_pts_uv = filter_invalid_points(next_pts_uv)

    import matplotlib.pyplot as plt
    def saveimg(i, imgs, uv, name):
        plt.imshow(imgs[i] / 256)
        plt.scatter(uv[i][:, 0], uv[i][:, 1], s=2)
        plt.savefig(f'{name}_{i}.jpg')
        plt.cla()
    for i in range(6):
        saveimg(i, curr_imgs, curr_pts_uv, 'curr')
        saveimg(i, prev_imgs, prev_pts_uv, 'prev')
        saveimg(i, next_imgs, next_pts_uv, 'next')
        
    pass
