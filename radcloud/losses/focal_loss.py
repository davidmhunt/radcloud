from torch.nn import Module
from torch import sigmoid
from torch import Tensor
import torch

#import existing loss functions
from torch.nn import BCEWithLogitsLoss

class FocalLoss(Module):

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 reduction: str = "none"):
        
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self,inputs:Tensor, targets:Tensor):

        p = sigmoid(inputs)

        ce_loss = self.bce_loss(inputs,targets)

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss