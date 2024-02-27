from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .rgb_loss_ms import RGBLossMS, SemLossMS, SemCELossMS
from .reproj_loss_mono_multi_new import ReprojLossMonoMultiNew
from .reproj_loss_mono_multi_new_combine import ReprojLossMonoMultiNewCombine
from .edge_loss_3d_ms import EdgeLoss3DMS
from .eikonal_loss import EikonalLoss
from .sparsity_loss import SparsityLoss, HardSparsityLoss, SoftSparsityLoss, AdaptiveSparsityLoss
from .second_grad_loss import SecondGradLoss
