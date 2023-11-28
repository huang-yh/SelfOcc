import torch, torch.nn as nn
import numpy as np


class RaySampler(nn.Module):

    def __init__(
            self,
            ray_sample_mode='fixed',    # fixed, cellular
            ray_number=[192, 400],      # 192 * 400
            ray_img_size=[768, 1600],
            ray_upper_crop=0,
            ray_x_dsr_max=None,
            ray_y_dsr_max=None):
        super().__init__()

        self.ray_sample_mode = ray_sample_mode
        self.ray_number = ray_number[0] * ray_number[1]
        self.ray_resize = ray_number
        self.ray_img_size = ray_img_size
        assert ray_sample_mode in ['fixed', 'cellular', 'random'] # TODO

        if ray_sample_mode == 'fixed':
            ray_x_dsr = 1.0 * ray_img_size[1] / ray_number[1]
            ray_y_dsr = 1.0 * ray_img_size[0] / ray_number[0]
            ray_x = torch.arange(ray_number[1], dtype=torch.float) * ray_x_dsr
            ray_y = torch.arange(ray_number[0], dtype=torch.float) * ray_y_dsr
            rays = torch.stack([
                ray_x.unsqueeze(0).expand(ray_number[0], -1),
                ray_y.unsqueeze(1).expand(-1, ray_number[1])], dim=-1).flatten(0, 1) # HW, 2
            self.register_buffer('rays', rays, False)
        elif ray_sample_mode == 'cellular':
            self.ray_upper_crop = ray_upper_crop
            self.ray_x_dsr_max = 1.0 * ray_img_size[1] / ray_number[1]
            self.ray_y_dsr_max = 1.0 * (ray_img_size[0] - ray_upper_crop) / ray_number[0]
            if ray_x_dsr_max is not None:
                self.ray_x_dsr_max = ray_x_dsr_max
            if ray_y_dsr_max is not None:
                self.ray_y_dsr_max = ray_y_dsr_max
            assert self.ray_x_dsr_max > 1 and self.ray_y_dsr_max > 1
            ray_x = torch.arange(ray_number[1], dtype=torch.float)
            ray_y = torch.arange(ray_number[0], dtype=torch.float)
            rays = torch.stack([
                ray_x.unsqueeze(0).expand(ray_number[0], -1),
                ray_y.unsqueeze(1).expand(-1, ray_number[1])], dim=-1) # H, W, 2
            self.register_buffer('rays', rays, False)

    def forward(self):
        device = self.rays.device
        
        if self.ray_sample_mode == 'fixed':
            return self.rays
        elif self.ray_sample_mode == 'random':
            rays = torch.rand(self.ray_number, 2, device=device)
            rays[:, 0] = rays[:, 0] * self.ray_img_size[1]
            rays[:, 1] = rays[:, 1] * self.ray_img_size[0]
            return rays
        elif self.ray_sample_mode == 'cellular':
            ray_x_dsr = np.random.uniform() * (self.ray_x_dsr_max - 1) + 1
            ray_y_dsr = np.random.uniform() * (self.ray_y_dsr_max - 1) + 1
            ray_x_emp_max = self.ray_img_size[1] - self.ray_resize[1] * ray_x_dsr
            ray_y_emp_max = self.ray_img_size[0] - self.ray_upper_crop - self.ray_resize[0] * ray_y_dsr
            ray_x_emp = np.random.uniform() * ray_x_emp_max
            ray_y_emp = np.random.uniform() * ray_y_emp_max
            rays = self.rays.clone() # H, W, 2
            rays[..., 0] = rays[..., 0] * ray_x_dsr + ray_x_emp
            rays[..., 1] = rays[..., 1] * ray_y_dsr + ray_y_emp + self.ray_upper_crop
            return rays.flatten(0, 1)
