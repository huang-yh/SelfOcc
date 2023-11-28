from mmseg.registry import MODELS
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import ModuleList
import torch.nn as nn, torch, copy
from torch.nn.init import normal_
from mmengine.logging import MMLogger
logger = MMLogger.get_instance('selfocc')

from ..base_encoder import BaseEncoder
from .utils import point_sampling
from .attention import BEVCrossAttention, BEVDeformableAttention

from .mappings import GridMeterMapping


@MODELS.register_module()
class BEVFormerEncoder(BaseEncoder):

    def __init__(
        self,
        mapping_args: dict,
        # bev_inner=128,
        # bev_outer=32,
        # range_inner=51.2,
        # range_outer=51.2,
        # nonlinear_mode='linear_upscale',
        # z_inner=20,
        # z_outer=10,
        # z_ranges=[-5.0, 3.0, 11.0],

        embed_dims=128,
        num_cams=6,
        num_feature_levels=4,
        positional_encoding=None,
        num_points_cross=32,
        num_points_self=16,
        transformerlayers=None, 
        num_layers=None,
        init_cfg=None):

        super().__init__(init_cfg)

        # self.bev_inner = bev_inner
        # self.bev_outer = bev_outer
        # self.range_inner = range_inner
        # self.range_outer = range_outer
        # assert nonlinear_mode == 'linear_upscale' # TODO
        # self.nonlinear_mode = nonlinear_mode
        # self.z_inner = z_inner
        # self.z_outer = z_outer
        # self.z_ranges = z_ranges
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.mapping = GridMeterMapping(
            # bev_inner,
            # bev_outer,
            # range_inner,
            # range_outer,
            # nonlinear_mode,
            # z_inner,
            # z_outer,
            # z_ranges
            **mapping_args)
        
        # bev_size = 2 * (bev_inner + bev_outer) + 1
        size_h = self.mapping.size_h
        size_w = self.mapping.size_w
        size_d = self.mapping.size_d
        bev_grid = torch.stack(
            [torch.arange(size_h, dtype=torch.float).unsqueeze(-1).expand(-1, size_w),
            torch.arange(size_w, dtype=torch.float).unsqueeze(0).expand(size_h, -1)], dim=-1)
        bev_meter = self.mapping.grid2meter(bev_grid)
        positional_encoding.update({'bev_meter': bev_meter})
        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.bev_size = [size_h, size_w]

        # transformer layers
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.pre_norm = self.layers[0].pre_norm
        logger.info('use pre_norm: ' + str(self.pre_norm))
        
        # other learnable embeddings
        self.level_embeds = nn.Parameter(
            torch.randn(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn(self.num_cams, self.embed_dims))

        # prepare reference points used in image cross-attention and cross-view hybrid-attention
        self.num_points_cross = num_points_cross
        self.num_points_self = num_points_self

        # uniform_d = torch.linspace(0, z_inner + z_outer, num_points_cross)
        uniform_d = torch.linspace(0, size_d - 1, num_points_cross)
        bev_3d_grid = torch.cat([
            bev_grid.unsqueeze(2).expand(-1, -1, num_points_cross, -1),
            uniform_d.reshape(1, 1, num_points_cross, 1).expand(size_h, size_w, -1, -1)
        ], dim=-1) # H, W, D, 3
        ref_3d = self.mapping.grid2meter(bev_3d_grid)
        self.register_buffer('ref_3d', ref_3d.flatten(0, 1).transpose(0, 1), False)
        
        bev_grid_normed = bev_grid.clone()
        bev_grid_normed[..., 0] = bev_grid_normed[..., 0] / (size_h - 1)
        bev_grid_normed[..., 1] = bev_grid_normed[..., 1] / (size_w - 1)
        self.register_buffer('ref_2d', bev_grid_normed, False) # H, W, 2
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, BEVCrossAttention) or \
                isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, BEVDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
    
    def forward_layers(
        self,
        bev_query, # b, c, h, w
        key,
        value,
        bev_pos=None, # b, h, w, c
        spatial_shapes=None,
        level_start_index=None,
        img_metas=None,
        **kwargs
    ):
        bs = bev_query.shape[0]

        ref_3d = self.ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1) # bs, p, HW, 3
        reference_points_cam, bev_mask = point_sampling(ref_3d, img_metas) # num_cam, bs, hw++, #p, 2
        
        ref_2d = self.ref_2d.unsqueeze(0).repeat(bs, 1, 1, 1) # bs, H, W, 2
        ref_2d = ref_2d.reshape(bs, -1, 1, 2) # bs, HW, 1, 2

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cam,
                bev_masks=bev_mask,
                bev_size=self.bev_size,
                **kwargs)
            bev_query = output

        return bev_query

    def forward(
        self,         
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        """Forward function.
        Args:
            img_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        img_feats = ms_img_feats
        bs = img_feats[0].shape[0]
        dtype = img_feats[0].dtype
        device = img_feats[0].device

        # bev queries and pos embeds
        bev_queries = representation # bs, HW, C
        bev_pos = self.positional_encoding().unsqueeze(0).repeat(bs, 1, 1) # bs, HW, C
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # forward layers
        bev_embed = self.forward_layers(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=metas,
        )
        
        return {'representation': bev_embed}

