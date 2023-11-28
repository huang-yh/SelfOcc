import torch
import torch.nn.functional as F

def multi2single_scale(
        ms_feats, imgs=None, 
        max_shape=None, 
        use_scales='all',
        normalize=None,
        scale_factor=None):

    num_level = len(ms_feats)
    if use_scales == 'all':
        scales = range(num_level)
    else:
        assert isinstance(use_scales, list)
        scales = use_scales
    used_feats = [ms_feats[i] for i in scales]
    if imgs is not None:
        used_feats.append(imgs)

    first2dims = used_feats[0].shape[:2]
    if max_shape is None:
        max_shape = used_feats[0].shape[-2:]
    
    reshaped = []
    for i, feat in enumerate(used_feats):
        if not all([s1 == s2 for s1, s2 in zip(feat.shape[-2:], max_shape)]):
            feat = F.interpolate(
                feat.flatten(0, 1), 
                size=max_shape, 
                mode='bilinear', 
                align_corners=True).unflatten(0, first2dims)
        if normalize is not None and (imgs is None or i < len(used_feats) - 1):
            feat = F.normalize(feat, p=normalize, dim=2)
        if scale_factor is not None and (imgs is None or i < len(used_feats) - 1):
            feat = feat * scale_factor
        reshaped.append(feat)
        
    return torch.cat(reshaped, dim=2)
