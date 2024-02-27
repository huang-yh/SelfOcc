import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@OPENOCC_LOSS.register_module()
class ReprojLossMonoMultiNewCombine(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ray_indices': 'ray_indices',
                'weights': 'weights',
                'ts': 'ts',
                'metas': 'metas',
                'ms_rays': 'ms_rays'}
        else:
            self.input_dict = input_dict
        self.no_ssim = kwargs.get('no_ssim', False)
        self.img_size = kwargs.get('img_size', [768, 1600])
        self.ray_resize = kwargs.get('ray_resize', None)
        self.no_automask = kwargs.get('no_automask', False)
        self.dims = kwargs.get('dims', 3)
        self.no_ssim = self.no_ssim or (self.ray_resize is None)
        if not self.no_ssim:
            self.ssim = SSIM()
        
        self.loss_func = self.reproj_loss
        self.iter_counter = 0
    
    def reproj_loss(
            self, 
            curr_imgs, 
            prev_imgs, 
            next_imgs, 
            ray_indices,
            weights,
            ts,
            metas, 
            ms_rays,
            deltas=None):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        device = ts[0].device
        bs, num_cams = curr_imgs.shape[:2]
        num_rays = ms_rays.shape[0]
        assert bs == 1

        # prepare transformation matrices
        img2prevImg, img2nextImg = [], []
        for meta in metas:
            img2prevImg.append(meta['img2prevImg'])
            img2nextImg.append(meta['img2nextImg'])
        
        def list2tensor(trans):
            if isinstance(trans[0], (np.ndarray, list)):
                trans = np.asarray(trans)
                trans = ts[0].new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
            else:
                trans = torch.stack(trans, dim=0)
            trans = trans.reshape(bs, num_cams, 1, 4, 4)
            return trans

        img2prevImg = list2tensor(img2prevImg)
        img2nextImg = list2tensor(img2nextImg)
        
        tot_loss = 0.
        for cam, (ray_idx, weight, t) in enumerate(zip(ray_indices, weights, ts)):

            rays = ms_rays[ray_idx]
            if deltas is not None:
                delta = deltas[cam].detach()
                eps = torch.finfo(delta.dtype).eps
                weight = weight.clone()
                weight[delta < eps] = 0.
                weight = weight / delta.clamp_min(eps)

            pixel_coords = torch.ones((bs, 1, len(rays), 4), device=device) # B, N, R, 4
            pixel_coords[..., :2] = rays.reshape(1, 1, -1, 2)
            pixel_coords[..., :3] *= t.reshape(1, 1, -1, 1)
            pixel_coords = pixel_coords.unsqueeze(-1)

            @torch.cuda.amp.autocast(enabled=False)
            def cal_pixel(trans, coords):
                trans = trans.float()
                coords = coords.float()
                eps = 1e-5
                pixel = torch.matmul(trans, coords).squeeze(-1) # bs, N, R, 4
                mask = pixel[..., 2] > 0
                pixel = pixel[..., :2] / torch.maximum(torch.ones_like(pixel[..., :1]) * eps, pixel[..., 2:3])
                mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] < self.img_size[1]) & \
                              (pixel[..., 1] > 0) & (pixel[..., 1] < self.img_size[0])
                return pixel, mask
            
            pixel_prev, prev_mask = cal_pixel(img2prevImg[:, cam:(cam+1), ...], pixel_coords) # bs, N, 1, R, 2
            pixel_prev = pixel_prev.unsqueeze(2) # B, N, 1, R, 2
            pixel_next, next_mask = cal_pixel(img2nextImg[:, cam:(cam+1), ...], pixel_coords)
            pixel_next = pixel_next.unsqueeze(2)
            
            def sample_pixel(pixel, imgs):
                # imgs: B, N, 3, H, W
                # pixel: B, N, 1, R, 2
                pixel = pixel.clone()
                pixel[..., 0] /= self.img_size[1]
                pixel[..., 1] /= self.img_size[0]
                pixel = 2 * pixel - 1
                pixel_rgb = F.grid_sample(
                    imgs.flatten(0, 1), 
                    pixel.flatten(0, 1), 
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True) # BN, 3, 1, R
                tmp_dim = pixel_rgb.shape[1]
                pixel_rgb = pixel_rgb.reshape(bs, 1, tmp_dim, pixel_rgb.shape[-1])
                pixel_rgb = pixel_rgb.permute(0, 1, 3, 2) # B, N, R, 3
                return pixel_rgb
            
            def get_diff(x1, x2):
                return torch.mean(torch.abs(x1 - x2), dim=-1)
            
            rgb_prev = sample_pixel(pixel_prev, prev_imgs[:, cam:(cam+1), ...])
            rgb_next = sample_pixel(pixel_next, next_imgs[:, cam:(cam+1), ...])
            pix_curr = ms_rays.reshape(1, 1, 1, num_rays, 2) # bs, N, 1, r, 2
            rgb_curr = sample_pixel(pix_curr, curr_imgs[:, cam:(cam+1), ...]) # B, N, r, 3
            rgb_curr_ = torch.gather(
                rgb_curr, dim=-2, 
                index=ray_idx.reshape(1, 1, -1, 1).repeat(1, 1, 1, self.dims)) # B, N, R, 3
            diff_prev = get_diff(rgb_curr_, rgb_prev) # B, N, R
            diff_next = get_diff(rgb_curr_, rgb_next)
            diff_prev[~prev_mask] = 0.
            diff_next[~next_mask] = 0.
            diff = diff_prev + diff_next
            cnt = prev_mask.to(torch.float) + next_mask.to(torch.float)
            general_mask = cnt > 0
            cnt = torch.clamp(cnt, 1.0)
            diff = diff / cnt # B, N, R

            weight = weight.clone()
            weight[~general_mask.flatten()] = 0.
            weight_sum = torch.zeros(num_rays, dtype=weight.dtype, device=device)
            weight_sum.index_add_(-1, ray_idx, weight)
            weight_sum = weight_sum.clamp_min(torch.finfo(weight.dtype).eps) # r
            weight_sum = torch.gather(weight_sum, -1, ray_idx) # R
            weight = weight / weight_sum # R

            l1_loss = torch.zeros(num_rays, dtype=diff.dtype, device=device)
            l1_loss.index_add_(-1, ray_idx, weight * diff.flatten()) # r
            prev_next_loss = l1_loss

            if not self.no_ssim:
                rgb_prev[~prev_mask] = 0.
                rgb_next[~next_mask] = 0.
                rgb_combine_ = rgb_prev + rgb_next
                rgb_combine_ = rgb_combine_ / cnt.unsqueeze(-1)
                rgb_combine = torch.zeros( # r, 3
                    num_rays, self.dims, dtype=rgb_combine_.dtype, device=device)
                rgb_combine.index_add_(
                    0, ray_idx, rgb_combine_.reshape(-1, self.dims) * weight.unsqueeze(-1))
                
                ssim_loss = self.ssim(
                    rgb_combine.reshape(1, *self.ray_resize, self.dims).permute(0, 3, 1, 2),
                    rgb_curr.reshape(1, *self.ray_resize, self.dims).permute(0, 3, 1, 2))
                ssim_loss = ssim_loss.mean(1).flatten()
                prev_next_loss = 0.15 * prev_next_loss + 0.85 * ssim_loss
            
            def compute_reprojection_loss(pred, target):
                # pred, target: B, N, r, 3
                abs_diff = torch.abs(target - pred)
                l1_loss = abs_diff.mean(-1).flatten() # r
                if self.no_ssim:
                    reprojection_loss = l1_loss
                else:
                    pred_reshape = pred.reshape(1, *self.ray_resize, self.dims).permute(0, 3, 1, 2)
                    target_reshape = target.reshape(1, *self.ray_resize, self.dims).permute(0, 3, 1, 2)
                    ssim_loss = self.ssim(pred_reshape, target_reshape).mean(1).flatten()
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                return reprojection_loss

            if not self.no_automask:
                target_prev = sample_pixel(pix_curr, prev_imgs[:, cam:(cam+1), ...])
                target_next = sample_pixel(pix_curr, next_imgs[:, cam:(cam+1), ...])

                mask_prev_proj_loss = compute_reprojection_loss(target_prev, rgb_curr) # r
                mask_next_proj_loss = compute_reprojection_loss(target_next, rgb_curr)

                ray_filter = torch.zeros(num_rays, dtype=weight.dtype, device=device)
                ray_filter.index_add_(0, ray_idx, general_mask.flatten().to(weight.dtype))
                ray_filter = ray_filter == 0
                prev_next_loss[ray_filter] = 1e3

                proj_loss = torch.stack([
                    prev_next_loss, 
                    mask_prev_proj_loss,
                    mask_next_proj_loss], dim=-1)
                proj_loss, _ = torch.min(proj_loss, dim=-1)

                if self.writer and self.iter_counter % 10 == 0:
                    mask = _ > 0
                    self.writer.add_scalar(f'masked/{cam}', mask.sum(), self.iter_counter)
            else:
                proj_loss = prev_next_loss
            
            proj_loss_avg = torch.mean(proj_loss)
            tot_loss = tot_loss + proj_loss_avg

        self.iter_counter += 1

        return tot_loss / num_cams