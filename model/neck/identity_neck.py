from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class IdentityNeck(BaseModule):

    def __init__(self, init_cfg = None):
        super().__init__(init_cfg)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        return inputs


if __name__ == "__main__":

    from mmengine import Config
    from mmseg.models import builder

    cfg = Config(dict(
        type='IdentityNeck',
    ))
    model = builder.build_neck(cfg)
    # model = UNet2D(out_feature=256, use_decoder=True)
    pass