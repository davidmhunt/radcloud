from torch.nn import Module
from torch import sigmoid
from torch import Tensor
import torch

#import existing loss functions
from torch.nn import BCEWithLogitsLoss


class DiceLoss(Module):

    def __init__(self, dice_smooth = 1.0):

        super().__init__()
        self.dice_smooth = dice_smooth

    def forward(self, inputs:Tensor, outputs:Tensor):

        #pass the inputs through a sigmoid
        inputs = sigmoid(inputs)

        #flatten the inputs and outputs
        inputs = inputs.view(-1)
        outputs = outputs.view(-1)

        intersection = (inputs * outputs).sum()
        dice = (2.0 * intersection + self.dice_smooth)/(inputs.sum() + outputs.sum() + self.dice_smooth)

        return 1 - dice