import torch, torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


def get_feat_from_meter(num_freqs, meter):
    freqs = torch.arange(-1, num_freqs - 1, dtype=torch.float)
    freqs = torch.pi * (2 ** freqs)
    meter_freq = meter.unsqueeze(-1) * freqs[None, None, None, ...]
    freq_feat = torch.stack([
        torch.sin(meter_freq),
        torch.cos(meter_freq)], dim=-1)
    freq_feat = freq_feat.flatten(-3).flatten(0, 1)
    return freq_feat

@MODELS.register_module()
class TPVPositionalEncoding(BaseModule):

    def __init__(
            self, 
            num_freqs, 
            embed_dims, 
            tpv_meters, 
            tot_range,
            init_cfg=None):
        super().__init__(init_cfg)

        assert isinstance(tot_range, list) and len(tot_range) == 6
        pc_range = tot_range
        hw_meter, zh_meter, wz_meter = tpv_meters
        
        hw_meter = hw_meter.clone()
        hw_meter[..., 0] = (hw_meter[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        hw_meter[..., 1] = (hw_meter[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        hw_freq_feat = get_feat_from_meter(num_freqs[0], hw_meter)

        zh_meter = zh_meter.clone()
        zh_meter[..., 0] = (zh_meter[..., 0] - pc_range[1]) / (pc_range[4] - pc_range[1])
        zh_meter[..., 1] = (zh_meter[..., 1] - pc_range[2]) / (pc_range[5] - pc_range[2])
        zh_freq_feat = get_feat_from_meter(num_freqs[1], zh_meter)

        wz_meter = wz_meter.clone()
        wz_meter[..., 0] = (wz_meter[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        wz_meter[..., 1] = (wz_meter[..., 1] - pc_range[2]) / (pc_range[5] - pc_range[2])
        wz_freq_feat = get_feat_from_meter(num_freqs[2], wz_meter)

        self.register_buffer('hw_freq_feat', hw_freq_feat, False)
        self.register_buffer('zh_freq_feat', zh_freq_feat, False)
        self.register_buffer('wz_freq_feat', wz_freq_feat, False)
        self.position_layer_hw = nn.Linear(4 * num_freqs[0], embed_dims)
        self.position_layer_zh = nn.Linear(4 * num_freqs[1], embed_dims)
        self.position_layer_wz = nn.Linear(4 * num_freqs[2], embed_dims)

    def forward(self):
        pos_hw = self.position_layer_hw(self.hw_freq_feat)
        pos_zh = self.position_layer_zh(self.zh_freq_feat)
        pos_wz = self.position_layer_wz(self.wz_freq_feat)
        return [pos_hw, pos_zh, pos_wz]  # H, W, C
    
if __name__ == "__main__":
    import os
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)))
    print(sys.path)
    from ..bevformer.mappings import GridMeterMapping
    from mmcv.cnn.bricks.transformer import build_positional_encoding

    bev_inner = 2
    bev_outer = 2

    m = GridMeterMapping(
        bev_inner=2,
        bev_outer=2,
        range_inner=2,
        range_outer=4,
        z_inner=2,
        z_outer=2,
        z_ranges=[-1., 1., 5.])

    bev_size = 2 * (bev_inner + bev_outer) + 1
    bev_grid = torch.stack(
        [torch.arange(bev_size, dtype=torch.float).unsqueeze(-1).expand(-1, bev_size),
        torch.arange(bev_size, dtype=torch.float).unsqueeze(0).expand(bev_size, -1)], dim=-1)
    bev_meter = m.grid2meter(bev_grid)
    positional_encoding = dict(
        type='BEVPositionalEncoding',
        num_freqs=3,
        embed_dims=32,
        tot_range=6
    )
    positional_encoding.update({'bev_meter': bev_meter})
    # positional encoding
    positional_encoding = build_positional_encoding(positional_encoding)
    
    pass