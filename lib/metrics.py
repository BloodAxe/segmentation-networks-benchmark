import torch

from torch.nn.modules.loss import _Loss
import numpy as np
import torch.nn.functional as F


class JaccardScore(_Loss):
    def __init__(self):
        super(JaccardScore, self).__init__()

    def forward(self, output, target):
        output = F.sigmoid(output)
        target = target.float()

        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        jac = intersection / (union - intersection + 1e-7)
        return jac

    def __str__(self):
        return 'JaccardScore'


class PixelAccuracy(_Loss):
    def __init__(self):
        super(PixelAccuracy, self).__init__()

    def forward(self, output, target):
        output = F.sigmoid(output) > 0.5
        target = target.byte()

        n_true = torch.eq(output, target)
        n_all = torch.numel(target)
        n_true = n_true.sum()
        if n_true == 0:
            return n_true

        return n_true.float() / n_all

    def __str__(self):
        return 'PixelAccuracy'
