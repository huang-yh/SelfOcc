import torch
import numpy as np


def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar, offset=0):
    # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
    # generate points for hw and level 1
    h_ranges = torch.linspace(offset, tpv_h-1+offset, tpv_h) / tpv_h
    w_ranges = torch.linspace(offset, tpv_w-1+offset, tpv_w) / tpv_w
    h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
    hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
    hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
    # generate points for hw and level 2
    z_ranges = torch.linspace(offset, tpv_z-1+offset, num_points_in_pillar[2]) / tpv_z # #p
    z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
    h_ranges = torch.linspace(offset, tpv_h-1+offset, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
    hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
    # generate points for hw and level 3
    z_ranges = torch.linspace(offset, tpv_z-1+offset, num_points_in_pillar[2]) / tpv_z # #p
    z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
    w_ranges = torch.linspace(offset, tpv_w-1+offset, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(1, -1, 1).expand(tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
    hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
    
    # generate points for zh and level 1
    w_ranges = torch.linspace(offset, tpv_w-1+offset, num_points_in_pillar[1]) / tpv_w
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
    h_ranges = torch.linspace(offset, tpv_h-1+offset, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
    zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
    # generate points for zh and level 2
    z_ranges = torch.linspace(offset, tpv_z-1+offset, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
    h_ranges = torch.linspace(offset, tpv_h-1+offset, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
    zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
    # generate points for zh and level 3
    w_ranges = torch.linspace(offset, tpv_w-1+offset, num_points_in_pillar[1]) / tpv_w
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
    z_ranges = torch.linspace(offset, tpv_z-1+offset, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
    zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

    # generate points for wz and level 1
    h_ranges = torch.linspace(offset, tpv_h-1+offset, num_points_in_pillar[0]) / tpv_h
    h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
    w_ranges = torch.linspace(offset, tpv_w-1+offset, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
    wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
    # generate points for wz and level 2
    h_ranges = torch.linspace(offset, tpv_h-1+offset, num_points_in_pillar[0]) / tpv_h
    h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
    z_ranges = torch.linspace(offset, tpv_z-1+offset, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
    wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
    # generate points for wz and level 3
    w_ranges = torch.linspace(offset, tpv_w-1+offset, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
    z_ranges = torch.linspace(offset, tpv_z-1+offset, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
    wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

    reference_points = torch.cat([
        torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
        torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
        torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
    ], dim=0) # hw+zh+wz, 3, #p, 2
    
    return reference_points
