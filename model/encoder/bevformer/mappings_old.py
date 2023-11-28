import torch
from typing_extensions import Literal


class GridMeterMapping:

    def __init__(
        self,
        bev_inner=128,
        bev_outer=32,
        range_inner=51.2,
        range_outer=51.2,
        nonlinear_mode: Literal['linear_upscale', 'linear'] = 'linear_upscale',
        z_inner=20,
        z_outer=10,
        z_ranges=[-5.0, 3.0, 11.0],
    ) -> None:
        self.bev_inner = bev_inner
        self.bev_outer = bev_outer
        self.range_inner = range_inner
        self.range_outer = range_outer
        # assert nonlinear_mode == 'linear_upscale' # TODO
        self.nonlinear_mode = nonlinear_mode
        self.z_inner = z_inner
        self.z_outer = z_outer
        self.z_ranges = z_ranges

        self.hw_unit = range_inner * 1.0 / bev_inner
        if nonlinear_mode == 'linear':
            assert bev_outer == 0 and range_outer == 0 
        else:
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
    
    def meter2grid(self, meter):
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

        return torch.cat([wh[..., 1:2], wh[..., 0:1], d], dim=-1)
        

if __name__ == "__main__":

    m = GridMeterMapping(
        bev_inner=2,
        bev_outer=2,
        range_inner=2,
        range_outer=4,
        z_inner=2,
        z_outer=2,
        z_ranges=[-1., 1., 5.])
    
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
