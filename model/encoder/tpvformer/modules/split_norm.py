from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import build_norm_layer
import torch.nn as nn


@MODELS.register_module()
class MultiPlaneNorm(BaseModule):

    def __init__(
            self,
            embed_dims=64,
            norm_cfg=dict(type='LN'), 
            init_cfg=None,
            **kwargs):
        super().__init__(init_cfg)
        self.embed_dim = embed_dims
        self.norm_cfg = norm_cfg
        
        self._init_layers()

    def _init_layers(self):
        self.norms = nn.ModuleList()
        for i in range(3):
            self.norms.append(
                build_norm_layer(self.norm_cfg, self.embed_dim)[1])

    def forward(self, tpv):
        outputs = []
        for i, plane in enumerate(tpv):
            outputs.append(self.norms[i](plane))
        return outputs