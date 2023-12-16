from .base_loss import BaseLoss
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class SecondGradLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'second_grad': 'second_grad',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.second_grad_loss
    
    def second_grad_loss(self, second_grad):
        return second_grad.abs().mean()