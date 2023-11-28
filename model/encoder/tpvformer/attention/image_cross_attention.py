
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmcv.cnn.bricks.transformer import build_attention


@MODELS.register_module()
class TPVCrossAttention(BaseModule):

    def __init__(
            self,
            embed_dims=256,
            num_cams=6,
            dropout=0.1, 
            init_cfg=None,
            batch_first=True,
            num_heads=16,
            num_levels=4,
            num_points=[64, 64, 8]):
        super().__init__(init_cfg)

        deformable_attn_config_hw = dict(
            type='BEVCrossAttention',
            embed_dims=embed_dims,
            num_cams=num_cams,
            dropout=dropout,
            batch_first=batch_first,
            deformable_attention=dict(
                type='BEVDeformableAttention',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points[2],
                dropout=dropout,
                batch_first=batch_first))
        self.attn_hw = build_attention(deformable_attn_config_hw)

        deformable_attn_config_zh = dict(
            type='BEVCrossAttention',
            embed_dims=embed_dims,
            num_cams=num_cams,
            dropout=dropout,
            batch_first=batch_first,
            deformable_attention=dict(
                type='BEVDeformableAttention',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points[1],
                dropout=dropout,
                batch_first=batch_first))
        self.attn_zh = build_attention(deformable_attn_config_zh)
        
        deformable_attn_config_wz = dict(
            type='BEVCrossAttention',
            embed_dims=embed_dims,
            num_cams=num_cams,
            dropout=dropout,
            batch_first=batch_first,
            deformable_attention=dict(
                type='BEVDeformableAttention',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points[0],
                dropout=dropout,
                batch_first=batch_first))
        self.attn_wz = build_attention(deformable_attn_config_wz)
        self.attns = [self.attn_hw, self.attn_zh, self.attn_wz]

    def forward(self,
                query,
                key,
                value,
                residual=None,
                spatial_shapes=None,
                reference_points_cams=None,
                tpv_masks=None,
                level_start_index=None,
                **kwargs):
        result = []

        for i in range(3):
            out = self.attns[i](
                query[i],
                key,
                value,
                residual[i] if residual is not None else None,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams[i],
                bev_masks=tpv_masks[i])
            result.append(out)

        return result
        
