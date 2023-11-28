from .base_lifter import BaseLifter
from mmseg.registry import MODELS
import torch, torch.nn as nn


@MODELS.register_module()
class BEVQueryLifter(BaseLifter):

    def __init__(
        self,
        bev_h,
        bev_w, 
        dim,
        init_cfg=None, 
        **kwargs) -> None:

        super().__init__(init_cfg, **kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.dim = dim
        self.bev = nn.Parameter(torch.randn(1, bev_h * bev_w, dim))

    def forward(self, ms_img_feats, *args, **kwargs):
        bs = ms_img_feats[0].shape[0]
        bev = self.bev.to(ms_img_feats[0].dtype).repeat(bs, 1, 1)
        return {'representation': bev}
    