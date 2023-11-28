import torch, torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmcv.cnn.bricks.transformer import build_feedforward_network


@MODELS.register_module()
class MultiPlaneFFN(BaseModule):

    def __init__(
            self,
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True),
            init_cfg=None):
        super().__init__(init_cfg)
        self.ffn_cfg = dict(
            type='FFN',
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=ffn_drop,
            act_cfg=act_cfg
        )
        self._init_layers()

    def _init_layers(self):
        self.ffns = nn.ModuleList()
        for i in range(3):
            self.ffns.append(
                build_feedforward_network(self.ffn_cfg))

    def forward(self, tpv, identity=None):
        outputs = []
        if identity is None:
            identity = [None] * 3
        for i, plane in enumerate(tpv):
            outputs.append(self.ffns[i](plane, identity[i]))
        return outputs