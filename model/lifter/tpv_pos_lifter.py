from .base_lifter import BaseLifter
from mmseg.registry import MODELS
import torch, torch.nn as nn
from model.encoder.bevformer.mappings import GridMeterMapping

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
class TPVPositionLifter(BaseLifter):

    def __init__(
        self,
        embed_dims,
        tot_range,
        num_freqs,
        mapping_args,
        init_cfg=None, 
        **kwargs) -> None:

        super().__init__(init_cfg, **kwargs)

        self.mapping = GridMeterMapping(**mapping_args)
        size_h = self.mapping.size_h
        size_w = self.mapping.size_w
        size_d = self.mapping.size_d

        hw_grid = torch.stack(
            [torch.arange(size_h, dtype=torch.float).unsqueeze(-1).expand(-1, size_w),
             torch.arange(size_w, dtype=torch.float).unsqueeze(0).expand(size_h, -1),
             torch.zeros(size_h, size_w)],
             dim=-1)
        hw_meter = self.mapping.grid2meter(hw_grid)[..., [0, 1]]
        zh_grid = torch.stack(
            [torch.arange(size_h, dtype=torch.float).unsqueeze(0).expand(size_d, -1),
             torch.zeros(size_d, size_h),
             torch.arange(size_d, dtype=torch.float).unsqueeze(-1).expand(-1, size_h)],
             dim=-1)
        zh_meter = self.mapping.grid2meter(zh_grid)[..., [1, 2]]
        wz_grid = torch.stack(
            [torch.zeros(size_w, size_d),
             torch.arange(size_w, dtype=torch.float).unsqueeze(-1).expand(-1, size_d),
             torch.arange(size_d, dtype=torch.float).unsqueeze(0).expand(size_w, -1)],
             dim=-1)
        wz_meter = self.mapping.grid2meter(wz_grid)[..., [0, 2]]

        assert isinstance(tot_range, list) and len(tot_range) == 6
        pc_range = tot_range
        
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


    def forward(self, ms_img_feats, *args, **kwargs):
        bs = ms_img_feats[0].shape[0]
        tpv_hw = self.position_layer_hw(self.hw_freq_feat).unsqueeze(0).repeat(bs, 1, 1)
        tpv_zh = self.position_layer_zh(self.zh_freq_feat).unsqueeze(0).repeat(bs, 1, 1)
        tpv_wz = self.position_layer_wz(self.wz_freq_feat).unsqueeze(0).repeat(bs, 1, 1)
        return {'representation': [tpv_hw, tpv_zh, tpv_wz]}
    