from .base_loss import BaseLoss
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class EikonalLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'eik_grad': 'eik_grad',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.eikonal_loss
    
    def eikonal_loss(self, eik_grad):
        grad_theta = eik_grad
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
        return eikonal_loss