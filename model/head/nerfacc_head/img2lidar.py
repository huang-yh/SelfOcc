import torch, torch.nn as nn
import numpy as np, os
from dataset.utils import get_rm


class Img2LiDAR(nn.Module):

    def __init__(
            self, 
            trans_kw,
            trans_kw_eval=None,
            novel_view=None):
        super().__init__()
        if not isinstance(trans_kw, list):
            trans_kw = [trans_kw]
            self.two_split = False
        else:
            assert trans_kw == ['img2lidar', 'temImg2lidar']
            self.two_split = True
        self.trans_kw = trans_kw
        self.trans_kw_eval = trans_kw if trans_kw_eval is None else trans_kw_eval
        self.novel_view = novel_view

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, metas, rays):
        rays = rays.float()
        # prepare img2lidar
        img2lidar = []
        # for key in self.trans_kw:
        #     for img_meta in metas:
        #         img2lidar.extend(img_meta[key])
        if os.environ.get('eval', 'false') == 'true':
            trans_kw = self.trans_kw_eval
        else:
            trans_kw = self.trans_kw
        for img_meta in metas:
            temp = []
            for key in trans_kw:
                temp.extend(img_meta[key])
            if isinstance(temp[0], (np.ndarray, list)):
                temp = np.asarray(temp)
            else:
                temp = torch.stack(temp, dim=0)
            img2lidar.append(temp)
        if isinstance(img2lidar[0], np.ndarray):
            img2lidar = np.asarray(img2lidar) # B, N, 4, 4
            img2lidar = rays.new_tensor(img2lidar)
        else:
            img2lidar = torch.stack(img2lidar, dim=0)

        if self.novel_view is not None:
            z_r = self.novel_view[3]
            rot_mat = rays.new_tensor(get_rm(z_r, 'z', True))
            img2lidar[..., :3, :3] = rot_mat.unsqueeze(0).unsqueeze(0) @ \
                img2lidar[..., :3, :3]
        
        origin = img2lidar[..., :3, 3] # B, N, 3
        if self.novel_view is not None:
            origin[..., 0] = origin[..., 0] + self.novel_view[0]
            origin[..., 1] = origin[..., 1] + self.novel_view[1]
            origin[..., 2] = origin[..., 2] + self.novel_view[2]

        rays = rays.reshape(1, 1, -1, 2)
        # origin = img2lidar[..., :3, 3] # B, N, 3
        rays_pad = torch.cat([
            rays, torch.ones_like(rays[..., :1])], dim=-1) # 1, 1, HW, 3
        direction = torch.matmul(
            img2lidar[..., :3, :3].unsqueeze(2),
            rays_pad.unsqueeze(-1)).squeeze(-1) # B, N, HW, 3
        return origin, direction
