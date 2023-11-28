import torch, torch.nn as nn
from mmengine.model import BaseModule
from ...encoder.bevformer.mappings import GridMeterMapping
import torch.nn.functional as F
from ..utils.sh_render import SHRender


class BEVNeRF(BaseModule):

    def __init__(
        self, 
        mapping_args: dict,
        # bev_inner=128,
        # bev_outer=32,
        # range_inner=51.2,
        # range_outer=51.2,
        # nonlinear_mode='linear_upscale',
        # z_inner=20,
        # z_outer=10,
        # z_ranges=[-5.0, 3.0, 11.0],

        # mlp decoder 
        embed_dims=128,
        color_dims=0,
        sem_dims=0,
        density_layers=2,

        sh_deg=2,
        sh_act='relu',
        tpv=False,
        init_cfg=None):

        super().__init__(init_cfg)
        
        self.mapping = GridMeterMapping(
            # bev_inner,
            # bev_outer,
            # range_inner,
            # range_outer,
            # nonlinear_mode,
            # z_inner,
            # z_outer,
            # z_ranges
            **mapping_args)
        
        self.embed_dims = embed_dims
        self.color_dims = color_dims
        self.sem_dims = sem_dims
        self.z_size = self.mapping.size_d
        self.h_size = self.mapping.size_h
        self.w_size = self.mapping.size_w
        self.bev_size = [self.mapping.size_h, self.mapping.size_w]
        self.density_layers = density_layers
        self.tpv = tpv
        self._init_layers()

        self.color_converter = SHRender
        self.sh_deg = sh_deg
        self.sh_act = sh_act
        self.density_color = None
    
    def _init_layers(self):
        density_net = []
        for i in range(self.density_layers - 1):
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, self.embed_dims)])
        if not self.tpv:
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, (1 + self.color_dims+ self.sem_dims) * self.z_size)])
        else:
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, 1 + self.color_dims + self.sem_dims)])
        density_net = nn.Sequential(*density_net)
        self.density_net = density_net

    @torch.cuda.amp.autocast(enabled=False)
    def pre_compute_density_color(self, bev):
        if not self.tpv:
            assert bev.dim() == 3
            # bev = bev.unflatten(1, (self.bev_size, self.bev_size))
            bev = bev.unflatten(1, self.bev_size)
            density_color = self.density_net(bev).reshape(*bev.shape[:-1], self.z_size, -1)
            density_color = density_color.permute(0, 4, 1, 2, 3) # bs, C, h, w, d
        else:
            tpv_hw, tpv_zh, tpv_wz = bev
            tpv_hw = tpv_hw.reshape(-1, self.h_size, self.w_size, 1, self.embed_dims)
            tpv_hw = tpv_hw.expand(-1, -1, -1, self.z_size, -1)

            tpv_zh = tpv_zh.reshape(-1, self.z_size, self.h_size, 1, self.embed_dims).permute(0, 2, 3, 1, 4)
            tpv_zh = tpv_zh.expand(-1, -1, self.w_size, -1, -1)

            tpv_wz = tpv_wz.reshape(-1, self.w_size, self.z_size, 1, self.embed_dims).permute(0, 3, 1, 2, 4)
            tpv_wz = tpv_wz.expand(-1, self.h_size, -1, -1, -1)

            tpv = tpv_hw + tpv_zh + tpv_wz
            density_color = self.density_net(tpv).permute(0, 4, 1, 2, 3)

        self.density_color = density_color
        # print(f'type of self.density_color: {self.density_color.dtype}')

    @torch.cuda.amp.autocast(enabled=False)
    def query_density(self, x):
        if self.density_color.dtype == torch.float16:
            x = x.half()
        grid = self.mapping.meter2grid(x, True)

        # grid[..., :2] = grid[..., :2] / (self.bev_size - 1)
        # grid[..., 2:] = grid[..., 2:] / (self.z_size - 1)
        grid = 2 * grid - 1
        grid = grid.reshape(1, -1, 1, 1, 3)

        density_color = F.grid_sample(
            self.density_color,
            grid[..., [2, 1, 0]],
            mode='bilinear',
            align_corners=True) # bs, c, n, 1, 1 
        
        density_color = density_color.permute(0, 2, 3, 4, 1).flatten(0, 3) # bs*n, c
        sigma = density_color[:, :1]
        return F.softplus(sigma) # F.relu(sigma)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, condition=None):
        if self.density_color.dtype == torch.float16:
            x = x.half()
            condition = condition.half() if condition is not None else None

        grid = self.mapping.meter2grid(x, True)

        # grid[..., :2] = grid[..., :2] / (self.bev_size - 1)
        # grid[..., 2:] = grid[..., 2:] / (self.z_size - 1)
        grid = 2 * grid - 1
        grid = grid.reshape(1, -1, 1, 1, 3)

        density_color = F.grid_sample(
            self.density_color,
            grid[..., [2, 1, 0]],
            mode='bilinear',
            align_corners=True) # bs, c, n, 1, 1 
        
        density_color = density_color.permute(0, 2, 3, 4, 1).flatten(0, 3) # bs*n, c
        sigma, sample_colors, sample_sems = density_color[:, :1], \
            density_color[:, 1:(1+self.color_dims)], density_color[:, (1+self.color_dims):]
        if self.color_dims > 0:
            sample_colors = self.color_converter(
                None, condition, sample_colors, self.sh_deg, self.sh_act)
            rgb = sample_colors.reshape(-1, 3)
        else:
            rgb = torch.empty((sigma.shape[0], 0), device=sigma.device, dtype=sigma.dtype)
        if self.sem_dims > 0:
            sems = torch.softmax(sample_sems, dim=-1)
        else:
            sems = torch.empty((sigma.shape[0], 0), device=sigma.device, dtype=sigma.dtype)
        return rgb, F.softplus(sigma), sems # F.relu(sigma)


    @torch.cuda.amp.autocast(enabled=False)
    def forward_geo(self, x):
        if self.density_color.dtype == torch.float16:
            x = x.half()

        grid = self.mapping.meter2grid(x, True)
        grid = 2 * grid - 1
        grid = grid.reshape(1, -1, 1, 1, 3)

        density_color = F.grid_sample(
            self.density_color,
            grid[..., [2, 1, 0]],
            mode='bilinear',
            align_corners=True) # bs, c, n, 1, 1 
        
        density_color = density_color.permute(0, 2, 3, 4, 1).flatten(0, 3) # bs*n, c
        sigma, sample_sems = density_color[:, :1], density_color[:, (1+self.color_dims):]
        if self.sem_dims > 0:
            sems = torch.softmax(sample_sems, dim=-1)
        else:
            sems = torch.empty((sigma.shape[0], 0), device=sigma.device, dtype=sigma.dtype)
        return F.softplus(sigma), sems # F.relu(sigma)
