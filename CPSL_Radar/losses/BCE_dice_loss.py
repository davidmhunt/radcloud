from torch.nn import Module
from torch import sigmoid
from torch import Tensor
import torch

#import existing loss functions
from torch.nn import BCEWithLogitsLoss
from CPSL_Radar.losses.dice_loss import DiceLoss

class BCE_DICE_Loss(Module):

    def __init__(self, dice_weight = 0.1, dice_smooth = 1.0):

        super().__init__()
        self.dice_weight = dice_weight
        
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(dice_smooth=dice_smooth)
    def forward(self, inputs:Tensor, outputs:Tensor):

        return self.bce_loss(inputs,outputs) + (self.dice_weight * self.dice_loss(inputs,outputs))