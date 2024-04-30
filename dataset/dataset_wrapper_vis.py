
import numpy as np, torch
from torch.utils import data
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    RandomScaleImageMultiViewImage
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
from . import OPENOCC_DATAWRAPPER

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_vis(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            scale_rate=1,
            img_norm_cfg=img_norm_cfg,
            pad_img_size=None,
            pad_scale_rate=None,
            **kwargs
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.scale_rate = scale_rate

        val_transforms = [
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)]
        if scale_rate != 1 or pad_scale_rate is not None:
            val_transforms.insert(1, RandomScaleImageMultiViewImage([scale_rate], pad_scale_rate=pad_scale_rate))
        self.transforms = val_transforms

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        return self.deal_with_length2_dataset(data)
        
    def deal_with_length2_dataset(self, data):
        input_imgs, img_metas = data
        img_metas['img_shape'] = input_imgs[0].shape[:2]

        # deal with img augmentations
        input_imgs, imgs_dict = forward_aug(input_imgs, img_metas, self.transforms)
        input_imgs = self.to_tensor(input_imgs)

        img_metas['scale_rate'] = self.scale_rate
        if 'focal_ratios_x' in imgs_dict:
            img_metas['focal_ratios_x'] = imgs_dict['focal_ratios_x']
        if 'focal_ratios_y' in imgs_dict:
            img_metas['focal_ratios_y'] = imgs_dict['focal_ratios_y']
        img_metas['flip'] = imgs_dict.get('flip', False)

        data_tuple = (input_imgs, 
                      img_metas,)
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
