import torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


@OPENOCC_LOSS.register_module()
class EdgeLoss3DMS(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'curr_imgs': 'curr_imgs',
                'ms_depths': 'ms_depths',
                'ms_rays': 'ms_rays'
            }
        else:
            self.input_dict = input_dict
        self.img_size = kwargs.get('img_size', [768, 1600])
        self.ray_resize = kwargs.get('ray_resize', None)
        self.use_inf_mask = kwargs.get('use_inf_mask', False)
        # self.inf_dist = kwargs.get('inf_dist', 1e6)
        assert self.ray_resize is not None
        self.loss_func = self.edge_loss
    
    def edge_loss(self, curr_imgs, ms_depths, ms_rays, ms_accs=None, max_depths=None):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        if self.use_inf_mask:
            assert ms_accs is not None and max_depths is not None
        if not isinstance(ms_rays, list):
            ms_rays = [ms_rays] * len(ms_depths)
        bs, num_cams, num_rays = ms_depths[0].shape

        tot_loss = 0.
        for scale, (depth, rays) in enumerate(zip(ms_depths, ms_rays)):
            pixel_curr = rays.clone().reshape(1, 1, num_rays, 2).repeat(
                bs * num_cams, 1, 1, 1) # bs*N, 1, R, 2
            pixel_curr[..., 0] /= self.img_size[1]
            pixel_curr[..., 1] /= self.img_size[0]
            pixel_curr = pixel_curr * 2 - 1
            rgb_curr = F.grid_sample(
                curr_imgs.flatten(0, 1), 
                pixel_curr, 
                mode='bilinear',
                padding_mode='border',
                align_corners=True) # bs*N, 3, 1, R
            rgb_curr = rgb_curr.reshape(bs * num_cams, -1, *self.ray_resize)

            if self.use_inf_mask:
                depth = depth * ms_accs[scale] + max_depths[scale] * (1 - ms_accs[scale])
                
            depth = depth.reshape(bs * num_cams, 1, *self.ray_resize)
            mean_depth = depth.mean(2, True).mean(3, True)
            norm_depth = depth / (mean_depth + 1e-6)
            smooth_loss = get_smooth_loss(norm_depth, rgb_curr)
            
            tot_loss += smooth_loss

        return tot_loss / len(ms_depths)
