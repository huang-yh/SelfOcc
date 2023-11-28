from .base_lifter import BaseLifter
from mmseg.registry import MODELS
import torch, torch.nn as nn


@MODELS.register_module()
class TPVQueryLifter(BaseLifter):

    def __init__(
        self,
        tpv_h,
        tpv_w,
        tpv_z, 
        dim,
        init_cfg=None, 
        **kwargs) -> None:

        super().__init__(init_cfg, **kwargs)
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.dim = dim
        self.tpv_hw = nn.Parameter(torch.randn(1, tpv_h * tpv_w, dim))
        self.tpv_zh = nn.Parameter(torch.randn(1, tpv_z * tpv_h, dim))
        self.tpv_wz = nn.Parameter(torch.randn(1, tpv_w * tpv_z, dim))

    def forward(self, ms_img_feats, *args, **kwargs):
        bs = ms_img_feats[0].shape[0]
        # dtype = ms_img_feats[0].dtype
        # tpv_hw = self.tpv_hw.to(dtype).repeat(bs, 1, 1)
        # tpv_zh = self.tpv_zh.to(dtype).repeat(bs, 1, 1)
        # tpv_wz = self.tpv_wz.to(dtype).repeat(bs, 1, 1)
        tpv_hw = self.tpv_hw.repeat(bs, 1, 1)
        tpv_zh = self.tpv_zh.repeat(bs, 1, 1)
        tpv_wz = self.tpv_wz.repeat(bs, 1, 1)
        return {'representation': [tpv_hw, tpv_zh, tpv_wz]}
    