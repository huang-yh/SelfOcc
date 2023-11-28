import os, numpy as np

from openseed.BaseModel import BaseModel
from openseed import build_model
from openseed.utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
import torch.nn.functional as F
import torch


def build_openseed_model():
    args = 'evaluate --conf_files config/openseed/openseed_swint_lang.yaml --overrides WEIGHT ckpts/openseed_model_state_dict_swint_51.2ap.pt'
    opt, cmdline_args = load_opt_command(args.split())
    pretrained_pth = os.path.join(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    stuff_classes = [
        'barrier',
        'bicycle',
        'bus',
        'car', # 'wagon', 'van', 'minivan', 'SUV', 'jeep',
        'construction_vehicle', 'crane',
        'motorcycle', # 'vespa', 'Scooter',
        'person', 
        'traffic_cone',
        'trailer', 'trailer_truck',
        'truck',
        'road', 
        'other_flat', # 'rail track', 'lake', 'river',
        'sidewalk', 
        'terrain', 'grass', # 'hill', # 'sand', 'gravel',
        'building', 'wall', # 'guard rail', 'fence', # 'pole',
        'tree', # 'plant', 
        'sky']
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )

    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    return model

def forward_openseed_model(model, img, size):
    img = F.interpolate(img, [512, int(1.0 * size[1] / size[0] * 512)], mode='bicubic', align_corners=True)
    batch_inputs = [{'image': img[i], 'height': size[0], 'width': size[1]} for i in range(img.shape[0])]
    with torch.no_grad():
        outputs = model.forward(batch_inputs, inference_task="sem_seg")
        sem_seg = [outputs[i]['sem_seg'].max(0)[1] for i in range(img.shape[0])]
        sem_seg = torch.stack(sem_seg, dim=0)
    torch.cuda.empty_cache()
    return sem_seg