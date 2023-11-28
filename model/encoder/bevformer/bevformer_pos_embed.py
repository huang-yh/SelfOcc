import torch, torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class BEVPositionalEncoding(BaseModule):

    def __init__(self, num_freqs, embed_dims, 
                 bev_meter, tot_range,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(tot_range, list):
            pc_range = [-1.0 * tot_range, -1.0 * tot_range, 0., tot_range, tot_range, 0.]
        else:
            pc_range = tot_range

        # bev_meter = (bev_meter + tot_range) / 2. / tot_range
        bev_meter[..., 0] = (bev_meter[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        bev_meter[..., 1] = (bev_meter[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        freqs = torch.arange(-1, num_freqs - 1, dtype=torch.float)
        freqs = torch.pi * (2 ** freqs)
        bev_meter_freq = bev_meter.unsqueeze(-1) * freqs[None, None, None, ...]
        freq_feat = torch.stack([
            torch.sin(bev_meter_freq),
            torch.cos(bev_meter_freq)], dim=-1)
        freq_feat = freq_feat.flatten(-3).flatten(0, 1)
        self.register_buffer('freq_feat', freq_feat, False)

        self.position_layer = nn.Linear(4 * num_freqs, embed_dims)

    def forward(self):
        pos = self.position_layer(self.freq_feat)
        return pos  # H, W, C
    
if __name__ == "__main__":
    import os
    import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)))
    print(sys.path)
    from mappings import GridMeterMapping
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