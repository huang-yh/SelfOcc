
import numpy as np, torch
from torch.utils import data
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage, \
    RandomFlip
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
from . import OPENOCC_DATAWRAPPER

# img_norm_cfg = dict(
    # mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_temporal(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
            scale_rate=1,
            photometric_aug=dict(
                use_swap_channel=False,
            ),
            use_temporal_aug=False,
            temporal_aug_list=[],
            img_norm_cfg=img_norm_cfg,
            supervision_img_size=None,
            supervision_scale_rate=None,
            use_flip=False,
            ref_focal_len=None,
            pad_img_size=None,
            random_scale=None,
            pad_scale_rate=None,
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.scale_rate = scale_rate
        self.use_temporal_aug = use_temporal_aug
        if use_temporal_aug:
            assert len(temporal_aug_list) > 0
        self.temporal_aug_list = temporal_aug_list

        photometric = PhotoMetricDistortionMultiViewImage(**photometric_aug)
        logger.info('using photometric augmentation: '+ str(photometric_aug))

        train_transforms = [
            photometric,
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        val_transforms = [
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        if scale_rate != 1 or ref_focal_len is not None or random_scale is not None or pad_scale_rate is not None:
            train_transforms.insert(2, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, random_scale, pad_scale_rate))
            val_transforms.insert(1, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, pad_scale_rate=pad_scale_rate))
        if use_flip:
            train_transforms.append(RandomFlip(0.5))
        
        if phase == 'train':
            self.transforms = train_transforms
        else:
            self.transforms = val_transforms
        if use_temporal_aug:
            self.temporal_transforms = [
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=4)]
            if supervision_scale_rate != 1:
                self.temporal_transforms.insert(1, RandomScaleImageMultiViewImage([supervision_scale_rate]))
        self.supervision_img_size = supervision_img_size

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            return self.deal_with_length2_dataset(data)
        elif len(data) == 3:
            return self.deal_with_length3_dataset(data)
    
    def deal_with_length3_dataset(self, data):
        input_imgs, anchor_imgs, img_metas = data
        img_metas['img_shape'] = input_imgs[0].shape[:2]

        # deal with img augmentations
        input_imgs, imgs_dict = forward_aug(input_imgs, img_metas, self.transforms)
        input_imgs = self.to_tensor(input_imgs)
        # img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['scale_rate'] = self.scale_rate
        # if 'focal_ratios' in imgs_dict:
        #     img_metas['focal_ratios'] = imgs_dict['focal_ratios']
        if 'focal_ratios_x' in imgs_dict:
            img_metas['focal_ratios_x'] = imgs_dict['focal_ratios_x']
        if 'focal_ratios_y' in imgs_dict:
            img_metas['focal_ratios_y'] = imgs_dict['focal_ratios_y']
        img_metas['flip'] = imgs_dict.get('flip', False)

        anchor_imgs = np.asarray(anchor_imgs).astype(np.float32)
        anchor_imgs = torch.from_numpy(anchor_imgs)
        anchor_imgs = anchor_imgs.permute(0, 1, 4, 2, 3)

        data_tuple = (input_imgs, 
                      anchor_imgs / 256.,
                      img_metas)
        return data_tuple
    
    def deal_with_length2_dataset(self, data):
        imgs, img_metas = data
        if len(imgs) == 4:
            input_imgs, curr_imgs, prev_imgs, next_imgs = imgs
            color_imgs = deepcopy(curr_imgs)
        elif len(imgs) == 3:
            curr_imgs, prev_imgs, next_imgs = imgs
            input_imgs = deepcopy(curr_imgs)
            color_imgs = deepcopy(curr_imgs)
        elif len(imgs) == 5:
            input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs = imgs
        img_metas['img_shape'] = input_imgs[0].shape[:2]

        # deal with img augmentations
        input_imgs, imgs_dict = forward_aug(input_imgs, img_metas, self.transforms)
        input_imgs = self.to_tensor(input_imgs)

        curr_aug = prev_aug = next_aug = None
        if 'curr_imgs' in self.temporal_aug_list:
            curr_aug, _ = forward_aug(curr_imgs, {}, self.temporal_transforms)
            curr_aug = self.to_tensor(curr_aug)
        if 'prev_imgs' in self.temporal_aug_list:
            prev_aug, _ = forward_aug(prev_imgs, {}, self.temporal_transforms)
            prev_aug = self.to_tensor(prev_aug)
        if 'next_imgs' in self.temporal_aug_list:
            next_aug, _ = forward_aug(next_imgs, {}, self.temporal_transforms)
            next_aug = self.to_tensor(next_aug)
            
        curr_imgs = self.to_tensor(curr_imgs)
        prev_imgs = self.to_tensor(prev_imgs)
        next_imgs = self.to_tensor(next_imgs)
        color_imgs = self.to_tensor(color_imgs)
        if self.supervision_img_size is not None:
            curr_imgs = F.interpolate(curr_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)
            prev_imgs = F.interpolate(prev_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)
            next_imgs = F.interpolate(next_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)

        # img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['scale_rate'] = self.scale_rate
        # if 'focal_ratios' in imgs_dict:
        #     img_metas['focal_ratios'] = imgs_dict['focal_ratios']
        if 'focal_ratios_x' in imgs_dict:
            img_metas['focal_ratios_x'] = imgs_dict['focal_ratios_x']
        if 'focal_ratios_y' in imgs_dict:
            img_metas['focal_ratios_y'] = imgs_dict['focal_ratios_y']
        img_metas['flip'] = imgs_dict.get('flip', False)

        data_tuple = (input_imgs, 
                      curr_imgs / 256., 
                      prev_imgs / 256., 
                      next_imgs / 256., 
                      color_imgs / 256.,
                      img_metas,
                      curr_aug,
                      prev_aug,
                      next_aug)
        return data_tuple


def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple

def forward_aug(imgs, metas, transforms):
    imgs_dict = {
        'img': imgs,
        'metas': metas,
    }
    for t in transforms:
        imgs_dict = t(imgs_dict)
    aug_imgs = imgs_dict['img']
    return aug_imgs, imgs_dict
