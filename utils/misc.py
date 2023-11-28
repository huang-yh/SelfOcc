import os.path as osp
import torch, numpy as np
import torch.nn.functional as F
from mmengine import Config
from mmengine.registry import MODELS
from mmengine.config.utils import MODULE2PACKAGE
from mmengine.infer import BaseInferencer
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model


def _load_model_from_metafile(model: str, scope: str = 'mmseg'):
    """Load config and weights from metafile.

    Args:
        model (str): model name defined in metafile.

    Returns:
        Tuple[Config, str]: Loaded Config and weights path defined in
        metafile.
    """
    init_default_scope(scope if scope else 'mmseg')
    model = model.lower()

    assert scope is not None, (
        'scope should be initialized if you want '
        'to load config from metafile.')
    assert scope in MODULE2PACKAGE, (
        f'{scope} not in {MODULE2PACKAGE}!,'
        'please pass a valid scope.')

    repo_or_mim_dir = BaseInferencer._get_repo_or_mim_dir(scope)
    for model_cfg in BaseInferencer._get_models_from_metafile(
            repo_or_mim_dir):
        model_name = model_cfg['Name'].lower()
        model_aliases = model_cfg.get('Alias', [])
        if isinstance(model_aliases, str):
            model_aliases = [model_aliases.lower()]
        else:
            model_aliases = [alias.lower() for alias in model_aliases]
        if (model_name == model or model in model_aliases):
            cfg = Config.fromfile(
                osp.join(repo_or_mim_dir, model_cfg['Config']))
            weights = model_cfg['Weights']
            weights = weights[0] if isinstance(weights, list) else weights
            return cfg, weights
    raise ValueError(f'Cannot find model: {model} in {scope}')

def _init_model(
    cfg,
    weights,
    device: str = 'cpu',
):
    """Initialize the model with the given config and checkpoint on the
    specific device.

    Args:
        cfg (ConfigType): Config containing the model information.
        weights (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. Defaults to 'cpu'.

    Returns:
        nn.Module: Model loaded with checkpoint.
    """
    checkpoint = None
    if weights is not None:
        checkpoint = _load_checkpoint(weights, map_location='cpu')

    # Delete the `pretrained` field to prevent model from loading the
    # the pretrained weights unnecessarily.
    if cfg.model.get('pretrained') is not None:
        del cfg.model.pretrained

    model = MODELS.build(cfg.model)
    model.cfg = cfg
    _load_checkpoint_to_model(model, checkpoint)
    model.to(device)
    model.eval()
    return model

nuscenes_to_cityscapes = [
    11, 13, 15, 15, 15, 15, 15, 15, 16, 14, 17, 7, 7, 4, 10, 3, 0, 6, 2
]

def generate_segmentation_map_2d(model_2d, inputs):
    bs, num_cams = inputs.shape[:2]
    outputs = model_2d(inputs.flatten(0, 1)) # bs*num_cams, c, h, w
    logits = torch.softmax(outputs, dim=1)
    nusc_logits = torch.zeros(
        (logits.shape[0], 18, *logits.shape[2:]), device=logits.device, dtype=logits.dtype)
    index = logits.new_tensor(nuscenes_to_cityscapes, dtype=torch.long)
    nusc_logits.index_add_(dim=1, index=index, source=logits)
    return nusc_logits.unflatten(0, (bs, num_cams)) # bs, num_cams, c, h, w

        # ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        #  'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        #  'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        #  'vegetation'],

nuscenes_to_openseed = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0
]
nuscenes_to_openseed = [
    1,
    2,
    3,
    4,
    5, 5,
    6,
    7,
    8,
    9, 9,
    10,
    11,
    12, 12, 12, 12,
    13,
    14, 14, 14,
    15, 15,
    16,
    0
]
def read_segmentation_map_2d(map_path, img_metas):
    labelss = []
    for img_meta in img_metas:

        token = img_meta['token']
        labels = []
        for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            label_path = osp.join(map_path, cam, token + '.npy')
            label = np.load(label_path).astype(np.int64)
            label = label[:768, :1600]
            label = torch.from_numpy(label).cuda()
            label = F.one_hot(label, num_classes=len(nuscenes_to_openseed)).float()
            labels.append(label.permute(2, 0, 1))
        labels = torch.stack(labels)
        labelss.append(labels)
    labelss = torch.stack(labelss) # b, n, c, h, w

    nusc_logits = torch.zeros(
        (*labelss.shape[:2], 17, *labelss.shape[3:]), device=labelss.device, dtype=labelss.dtype)
    index = labelss.new_tensor(nuscenes_to_openseed, dtype=torch.long)
    nusc_logits.index_add_(dim=2, index=index, source=labelss)
    return nusc_logits




    