import torch
from torch import nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W) in [0, 1]
        """

        # horizontal gradient
        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # vertical gradient
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.l1_loss(grad_pred_x, grad_target_x, reduction=self.reduction)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y, reduction=self.reduction)

        return loss_x + loss_y

def loss_fn1():
    loss_fn = torch.nn.L1Loss(reduction='mean') # L1 performs better than L2
    # loss_fn = torch.nn.MSELoss(reduction='mean')

    return loss_fn

if __name__ == '__main__':
    inp = torch.randn(1, 3, 224, 224)
    out = torch.randn(1, 3, 224, 224)

    loss1 = loss_fn1()(inp, inp)
    loss2 = loss_fn1()(inp, out)
    print(f'Case1: Same values Loss = {loss1}')
    print(f'Case2: Different values Loss = {loss2}')