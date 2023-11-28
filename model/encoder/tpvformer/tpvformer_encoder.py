from mmseg.registry import MODELS
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import ModuleList
import torch.nn as nn, torch, copy
from torch.nn.init import normal_
from mmengine.logging import MMLogger
logger = MMLogger.get_instance('selfocc')

from ..base_encoder import BaseEncoder
from ..bevformer.utils import point_sampling
from .utils import get_cross_view_ref_points
from ..bevformer.mappings import GridMeterMapping
from ..bevformer.attention import BEVCrossAttention, BEVDeformableAttention
from .attention import TPVCrossAttention, CrossViewHybridAttention
from .modules import CameraAwareSE


@MODELS.register_module()
class TPVFormerEncoder(BaseEncoder):

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
        num_points_cross=[64, 64, 8],
        num_points_self=[16, 16, 16],
        transformerlayers=None, 
        num_layers=None,
        camera_aware=False,
        camera_aware_mid_channels=None,
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
        self.camera_aware = camera_aware
        if camera_aware:
            if camera_aware_mid_channels is None:
                camera_aware_mid_channels = embed_dims
            self.camera_se_net = CameraAwareSE(
                embed_dims,
                camera_aware_mid_channels,
                embed_dims)

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

        positional_encoding.update({'tpv_meters': [hw_meter, zh_meter, wz_meter]})
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.tpv_size = [size_h, size_w, size_d]

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

        uniform_d = torch.linspace(0, size_d - 1, num_points_cross[2])
        hw_3d_grid = torch.cat([
            hw_grid[..., [0, 1]].unsqueeze(2).expand(-1, -1, num_points_cross[2], -1),
            uniform_d.reshape(1, 1, -1, 1).expand(size_h, size_w, -1, -1)], dim=-1)
        ref_3d_hw = self.mapping.grid2meter(hw_3d_grid) # H, W, P0, 3

        uniform_w = torch.linspace(0, size_w - 1, num_points_cross[1])
        zh_3d_grid = torch.cat([
            zh_grid[..., :1].unsqueeze(2).expand(-1, -1, num_points_cross[1], -1),
            uniform_w.reshape(1, 1, -1, 1).expand(size_d, size_h, -1, -1),
            zh_grid[..., 2:].unsqueeze(2).expand(-1, -1, num_points_cross[1], -1)
        ], dim=-1)
        ref_3d_zh = self.mapping.grid2meter(zh_3d_grid) # Z, H, P1, 3

        uniform_h = torch.linspace(0, size_h - 1, num_points_cross[0])
        wz_3d_grid = torch.cat([
            uniform_h.reshape(1, 1, -1, 1).expand(size_w, size_d, -1, -1),
            wz_grid[..., [1, 2]].unsqueeze(2).expand(-1, -1, num_points_cross[0], -1)
        ], dim=-1)
        ref_3d_wz = self.mapping.grid2meter(wz_3d_grid) # W, Z, P2, 3

        self.register_buffer('ref_3d_hw', ref_3d_hw.flatten(0, 1).transpose(0, 1), False)
        self.register_buffer('ref_3d_zh', ref_3d_zh.flatten(0, 1).transpose(0, 1), False)
        self.register_buffer('ref_3d_wz', ref_3d_wz.flatten(0, 1).transpose(0, 1), False)
        
        cross_view_ref_points = get_cross_view_ref_points(size_h, size_w, size_d, num_points_self)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points, False)
        # hw_grid_normed = hw_grid[..., [0, 1]].clone()
        # hw_grid_normed[..., 0] = hw_grid_normed[..., 0] / (size_h - 1)
        # hw_grid_normed[..., 1] = hw_grid_normed[..., 1] / (size_w - 1)

        # zh_grid_normed = zh_grid[..., [2, 0]].clone()
        # zh_grid_normed[..., 0] = zh_grid_normed[..., 0] / (size_d - 1)
        # zh_grid_normed[..., 1] = zh_grid_normed[..., 1] / (size_h - 1)

        # wz_grid_normed = wz_grid[..., [1, 2]].clone()
        # wz_grid_normed[..., 0] = wz_grid_normed[..., 0] / (size_w - 1)
        # wz_grid_normed[..., 1] = wz_grid_normed[..., 1] / (size_d - 1)

        # self.register_buffer('ref_2d_hw', hw_grid_normed, False) # H, W, 2
        # self.register_buffer('ref_2d_zh', zh_grid_normed, False) # H, W, 2
        # self.register_buffer('ref_2d_wz', wz_grid_normed, False) # H, W, 2
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, BEVCrossAttention) or \
                isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, BEVDeformableAttention) or \
                        isinstance(m, TPVCrossAttention) or \
                            isinstance(m, CrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
    
    def forward_layers(
        self,
        tpv_query, # b, c, h, w
        key,
        value,
        tpv_pos=None, # b, h, w, c
        spatial_shapes=None,
        level_start_index=None,
        img_metas=None,
        **kwargs
    ):
        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        for ref_3d in [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]:
            reference_points_cam, tpv_mask = point_sampling(
                ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1), img_metas)
            reference_points_cams.append(reference_points_cam) # num_cam, bs, hw++, #p, 2
            tpv_masks.append(tpv_mask)
        
        # ref_2d = self.ref_2d.unsqueeze(0).repeat(bs, 1, 1, 1) # bs, H, W, 2
        # ref_2d = ref_2d.reshape(bs, -1, 1, 2) # bs, HW, 1, 2
        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(
            0).expand(bs, -1, -1, -1, -1) # bs, hw++, 3, #p, 2

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                tpv_size=self.tpv_size,
                **kwargs)
            tpv_query = output

        return tpv_query

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
        tpv_queries = representation # bs, HW, C
        tpv_pos = self.positional_encoding()
        tpv_pos = [pos.unsqueeze(0).repeat(bs, 1, 1) for pos in tpv_pos]
        
        # add camera awareness if required
        if self.camera_aware:
            img_feats = self.camera_se_net(img_feats, metas)
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :]#.to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :]#.to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # forward layers
        tpv_embed = self.forward_layers(
            tpv_queries,
            feat_flatten,
            feat_flatten,
            tpv_pos=tpv_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=metas,
        )
        
        return {'representation': tpv_embed}

