import torch
from typing_extensions import Literal

class LinearMapping:

    def __init__(
            self,
            h_size=[128, 32],
            h_range=[51.2, 28.8],
            h_half=False,
            w_size=[128, 32],
            w_range=[51.2, 28.8],
            w_half=False,
            d_size=[20, 10],
            d_range=[-4.0, 4.0, 12.0]):
        
        self.h_size = h_size
        self.h_range = h_range
        self.h_half = h_half

        self.w_size = w_size
        self.w_range = w_range
        self.w_half = w_half

        self.d_size = d_size
        self.d_range = [d_range[1] - d_range[0], d_range[2] - d_range[1]]
        self.d_start = d_range[0]

        if h_half:
            self.h_tot_len = 1 + self.h_size[0] + self.h_size[1]
        else:
            self.h_tot_len = 1 + 2 * (self.h_size[0] + self.h_size[1])
        if w_half:
            self.w_tot_len = 1 + self.w_size[0] + self.w_size[1]
        else:
            self.w_tot_len = 1 + 2 * (self.w_size[0] + self.w_size[1])
        self.d_tot_len = 1 + self.d_size[0] + self.d_size[1]

    def grid2meter(self, grid):
        # grid: [..., (h, w, d)]
        h, w = grid[..., 0], grid[..., 1]
        if grid.shape[-1] == 3:
            d = grid[..., 2]
        else:
            d = None
        
        ## deal with h
        if not self.h_half:
            h_ctr = h - (self.h_size[0] + self.h_size[1])
        else:
            h_ctr = h
        h_abs = torch.abs(h_ctr)
        if self.h_size[1] == 0:
            y_abs = h_abs / self.h_size[0] * self.h_range[0]
        else:
            y_abs = torch.where(
                h_abs > self.h_size[0],
                self.h_range[0] + (h_abs - self.h_size[0]) / \
                    self.h_size[1] * self.h_range[1],
                h_abs / self.h_size[0] * self.h_range[0])
        y = torch.sign(h_ctr) * y_abs

        ## deal with w
        if not self.w_half:
            w_ctr = w - (self.w_size[0] + self.w_size[1])
        else:
            w_ctr = w
        w_abs = torch.abs(w_ctr)
        if self.w_size[1] == 0:
            x_abs = w_abs / self.w_size[0] * self.w_range[0]
        else:
            x_abs = torch.where(
                w_abs > self.w_size[0],
                self.w_range[0] + (w_abs - self.w_size[0]) / \
                    self.w_size[1] * self.w_range[1],
                w_abs / self.w_size[0] * self.w_range[0])
        x = torch.sign(w_ctr) * x_abs

        ## deal with d
        if d is not None:
            d_ctr = d
            d_abs = torch.abs(d_ctr)
            if self.d_size[1] == 0:
                z_abs = d_abs / self.d_size[0] * self.d_range[0]
            else:
                z_abs = torch.where(
                    d_abs > self.d_size[0],
                    self.d_range[0] + (d_abs - self.d_size[0]) / \
                        self.d_size[1] * self.d_range[1],
                    d_abs / self.d_size[0] * self.d_range[0])
            z = torch.sign(d_ctr) * z_abs + self.d_start

            return torch.stack([x, y, z], dim=-1)
        else:
            return torch.stack([x, y], dim=-1)

    def meter2grid(self, meter, normalize=False):
        x, y, z = meter[..., 0], meter[..., 1], meter[..., 2]

        ## deal with x
        x_abs = torch.abs(x)
        if self.w_size[1] == 0:
            w_abs = x_abs / self.w_range[0] * self.w_size[0]
        else:
            w_abs = torch.where(
                x_abs > self.w_range[0],
                self.w_size[0] + (x_abs - self.w_range[0]) / \
                    self.w_range[1] * self.w_size[1],
                x_abs / self.w_range[0] * self.w_size[0])
        w_ctr = torch.sign(x) * w_abs
        if not self.w_half:
            w = w_ctr + self.w_size[0] + self.w_size[1]
        else:
            w = w_ctr
        
        ## deal with y
        y_abs = torch.abs(y)
        if self.h_size[1] == 0:
            h_abs = y_abs / self.h_range[0] * self.h_size[0]
        else:
            h_abs = torch.where(
                y_abs > self.h_range[0],
                self.h_size[0] + (y_abs - self.h_range[0]) / \
                    self.h_range[1] * self.h_size[1],
                y_abs / self.h_range[0] * self.h_size[0])
        h_ctr = torch.sign(y) * h_abs
        if not self.h_half:
            h = h_ctr + self.h_size[0] + self.h_size[1]
        else:
            h = h_ctr
        
        ## deal with z
        z_ctr = z - self.d_start
        z_abs = torch.abs(z_ctr)
        if self.d_size[1] == 0:
            d_abs = z_abs / self.d_range[0] * self.d_size[0]
        else:
            d_abs = torch.where(
                z_abs > self.d_range[0],
                self.d_size[0] + (z_abs - self.d_range[0]) / \
                    self.d_range[1] * self.d_size[1],
                z_abs / self.d_range[0] * self.d_size[0])
        d = torch.sign(z_ctr) * d_abs

        if normalize:
            h = h / (self.h_tot_len - 1)
            w = w / (self.w_tot_len - 1)
            d = d / (self.d_tot_len - 1)
        
        return torch.stack([h, w, d], dim=-1)


class GridMeterMapping:

    def __init__(
        self,
        nonlinear_mode: Literal['linear_upscale', 'linear'] = 'linear_upscale',
        h_size=[128, 32],
        h_range=[51.2, 28.8],
        h_half=False,
        w_size=[128, 32],
        w_range=[51.2, 28.8],
        w_half=False,
        d_size=[20, 10],
        d_range=[-4.0, 4.0, 12.0]
    ) -> None:
        self.nonlinear_mode = nonlinear_mode
        if nonlinear_mode == 'linear_upscale':
            assert all([h == w for h, w in zip(h_size, w_size)])
            assert all([h == w for h, w in zip(h_range, w_range)])
            assert (not h_half) and (not w_half)
            self.mapping = NonLinearMapping(
                h_size[0],
                h_size[1],
                h_range[0],
                h_range[1],
                d_size[0],
                d_size[1],
                d_range)
            self.size_h = self.size_w = self.mapping.bev_size
            self.size_d = self.mapping.z_size
        elif nonlinear_mode == 'linear':
            self.mapping = LinearMapping(
                h_size,
                h_range,
                h_half,
                w_size,
                w_range,
                w_half,
                d_size,
                d_range)
            self.size_h = self.mapping.h_tot_len
            self.size_w = self.mapping.w_tot_len
            self.size_d = self.mapping.d_tot_len
        self.grid2meter = self.mapping.grid2meter
        self.meter2grid = self.mapping.meter2grid


class NonLinearMapping:

    def __init__(
        self,
        bev_inner=128,
        bev_outer=32,
        range_inner=51.2,
        range_outer=51.2,
        z_inner=20,
        z_outer=10,
        z_ranges=[-5.0, 3.0, 11.0],
    ) -> None:
        self.bev_inner = bev_inner
        self.bev_outer = bev_outer
        self.range_inner = range_inner
        self.range_outer = range_outer
        self.z_inner = z_inner
        self.z_outer = z_outer
        self.z_ranges = z_ranges
        self.bev_size = 1 + 2 * (self.bev_inner + self.bev_outer)
        self.z_size = 1 + self.z_inner + self.z_outer

        self.hw_unit = range_inner * 1.0 / bev_inner
        self.increase_unit = (range_outer - bev_outer * self.hw_unit) * 2.0 / bev_outer / (bev_outer + 1)
        
        self.z_unit = (z_ranges[1] - z_ranges[0]) * 1.0 / z_inner
        self.z_increase_unit = (z_ranges[2] - z_ranges[1] - z_outer * self.z_unit) * 2.0 / z_outer / (z_outer + 1)

    def grid2meter(self, grid):
        hw = grid[..., :2]
        hw_center = hw - (self.bev_inner + self.bev_outer)
        hw_center_abs = torch.abs(hw_center)
        yx_base_abs = hw_center_abs * self.hw_unit
        hw_outer = torch.relu(hw_center_abs - self.bev_inner)
        hw_outer_int = torch.floor(hw_outer)
        yx_outer_base = hw_outer_int * (hw_outer_int + 1) / 2.0 * self.increase_unit
        yx_outer_resi = (hw_outer - hw_outer_int) * (hw_outer_int + 1) * self.increase_unit
        yx_abs = yx_base_abs + yx_outer_base + yx_outer_resi
        yx = torch.sign(hw_center) * yx_abs

        if grid.shape[-1] == 3:
            d = grid[..., 2:3]
            d_center = d
            z_base = d_center * self.z_unit

            d_outer = torch.relu(d_center - self.z_inner)
            d_outer_int = torch.floor(d_outer)
            z_outer_base = d_outer_int * (d_outer_int + 1) / 2.0 * self.z_increase_unit
            z_outer_resi = (d_outer - d_outer_int) * (d_outer_int + 1) * self.z_increase_unit
            z = z_base + z_outer_base + z_outer_resi + self.z_ranges[0]
            
            return torch.cat([yx[..., 1:2], yx[..., 0:1], z], dim=-1)
        else:
            return yx[..., [1, 0]]
    
    def meter2grid(self, meter, normalize=False):
        xy = meter[..., :2]
        xy_abs = torch.abs(xy)
        wh_base_abs = xy_abs / self.hw_unit
        wh_base_abs = wh_base_abs.clamp_(max=self.bev_inner)
        xy_outer_abs = torch.relu(xy_abs - self.range_inner)

        wh_outer_base = torch.sqrt((1. / 2 + self.hw_unit / self.increase_unit) ** 2 + \
                                   2 * xy_outer_abs / self.increase_unit) - (1. / 2 + self.hw_unit / self.increase_unit)
        wh_outer_base = torch.floor(wh_outer_base)
        xy_outer_resi = xy_outer_abs - wh_outer_base * self.hw_unit - self.increase_unit * wh_outer_base * (wh_outer_base + 1) / 2
        wh_outer_resi = xy_outer_resi / (self.hw_unit + (wh_outer_base + 1) * self.increase_unit)
        wh_center_abs = wh_base_abs + wh_outer_base + wh_outer_resi
        wh_center = torch.sign(xy) * wh_center_abs
        wh = wh_center + self.bev_inner + self.bev_outer

        z = meter[..., 2:3]
        z_abs = z - self.z_ranges[0]
        d_base = z_abs / self.z_unit
        d_base = d_base.clamp_(max=self.z_inner)
        z_outer = torch.relu(z_abs - (self.z_ranges[1] - self.z_ranges[0]))

        d_outer_base = torch.sqrt((1. / 2 + self.z_unit / self.z_increase_unit) ** 2 + \
                                  2 * z_outer / self.z_increase_unit) - (1. / 2 + self.z_unit / self.z_increase_unit)
        d_outer_base = torch.floor(d_outer_base)
        z_outer_resi = z_outer - d_outer_base * self.z_unit - self.z_increase_unit * d_outer_base * (d_outer_base + 1) / 2
        d_outer_resi = z_outer_resi / (self.z_unit + (d_outer_base + 1) * self.z_increase_unit)
        d = d_base + d_outer_base + d_outer_resi

        if normalize:
            wh = wh / (self.bev_size - 1)
            d = d / (self.z_size - 1)

        return torch.cat([wh[..., 1:2], wh[..., 0:1], d], dim=-1)
        

if __name__ == "__main__":

    # m = GridMeterMapping(
    #     bev_inner=2,
    #     bev_outer=2,
    #     range_inner=2,
    #     range_outer=4,
    #     z_inner=2,
    #     z_outer=2,
    #     z_ranges=[-1., 1., 5.])
    m = GridMeterMapping(
        nonlinear_mode='linear',
        h_size=[2, 2],
        h_range=[2, 4],
        h_half=False,
        w_size=[2, 2],
        w_range=[2, 4],
        w_half=False,
        d_size=[2, 2],
        d_range=[-1., 1., 5.]
    )
    
    grid = [
        [4, 0, 0],
        [0, 4, 1],
        [4, 4, 2],
        [5, 6, 4],
        [1, 0, 1.5],
        [7.5, 8, 2.5]]
    
    print(m.grid2meter(torch.tensor(grid)))

    meter = [[-6.0000,  0.0000, -1.0000],
        [ 0.0000, -6.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000],
        [ 2.0000,  1.0000,  5.0000],
        [-6.0000, -3.6667,  0.5000],
        [ 6.0000,  4.8333,  1.8333]]
    
    print(m.meter2grid(torch.tensor(meter)))

    pass
