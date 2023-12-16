from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch, numpy as np


@OPENOCC_LOSS.register_module()
class SparsityLoss(BaseLoss):
    """
    StreetSurf style.
    """

    def __init__(self, weight=1.0, scale=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'density': 'density',
            }
        else:
            self.input_dict = input_dict
        self.scale = scale
        self.loss_func = self.sparsity_loss
    
    def sparsity_loss(self, density):
        loss = 1.0 / torch.cosh(density / (2.0 * self.scale))
        loss = torch.pow(loss, 2).mean()
        return loss
    
@OPENOCC_LOSS.register_module()
class HardSparsityLoss(BaseLoss):

    def __init__(self, weight=1.0, scale=1.0, thresh=0.2, crop=[[0, 0], [0, 0], [0, 0]], input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'density': 'density',
            }
        else:
            self.input_dict = input_dict
        self.scale = scale
        self.thresh = thresh
        self.crop = np.asarray(crop)
        self.loss_func = self.hard_sparsity_loss

    def hard_sigmoid(self, density):
        return torch.sigmoid((-1) * self.scale * density)
    
    def hard_sparsity_loss(self, density):
        if self.crop[0, 0] > 0:
            density[:self.crop[0, 0], ...] = 100
        if self.crop[0, 1] > 0:
            density[-self.crop[0, 1]:, ...] = 100
        if self.crop[1, 0] > 0:
            density[:, :self.crop[1, 0], :] = 100
        if self.crop[1, 1] > 0:
            density[:, -self.crop[1, 1]:, :] = 100
        if self.crop[2, 0] > 0:
            density[..., :self.crop[2, 0]] = 100
        if self.crop[2, 1] > 0:
            density[..., -self.crop[2, 1]:] = 100
        num_occupied_voxels = self.hard_sigmoid(density).mean()
        return torch.relu(num_occupied_voxels - self.thresh)


@OPENOCC_LOSS.register_module()
class SoftSparsityLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'density': 'density',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.soft_sparsity_loss
    
    def soft_sparsity_loss(self, density):
        return torch.relu(-1 * density).mean()


@OPENOCC_LOSS.register_module()
class AdaptiveSparsityLoss(BaseLoss):

    def __init__(self, weight=1, input_dict=None, slack=4.0, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'sdfs': 'sdfs',
                'ts': 'ts',
                'ms_depths': 'ms_depths'
            }
        else:
            self.input_dict = input_dict

        self.slack = slack
        self.loss_func = self.adaptive_sparsity_loss

    def adaptive_sparsity_loss(self, sdfs, ts, ms_depths):
        # ts: [R] * 6
        # sdfs: [R] * 6
        # ms_depths: B, N, R
        depths = ms_depths[0]
        bs, num_cams, num_rays = depths.shape
        assert bs == 1
        ts = torch.stack(ts, dim=0).reshape(bs, num_cams, num_rays, -1)
        sdfs = torch.stack(sdfs, dim=0).reshape(bs, num_cams, num_rays, -1)
        mask = ts > (depths + self.slack).unsqueeze(-1)

        sdfs_behind = sdfs[mask]
        return torch.relu(-1 * sdfs_behind).mean()