import torch.nn as nn
import torch.nn.functional as F
import torch
from apptest.util.utils import MASK_VALUE


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,
                 weight=None,
                 gamma=2,
                 reduction='mean',
                 ignore_index=MASK_VALUE):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input,
                                  target,
                                  reduction=self.reduction,
                                  weight=self.weight,
                                  ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt)**self.gamma * ce_loss).mean()
        return focal_loss