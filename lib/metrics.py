import torch
from torch import Tensor

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


class PRCurve(object):

    def __init__(self, thresholds=127):
        self.thresholds = np.arange(0., 1., 1. / thresholds, dtype=np.float32)

    def reset(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, y_true: Tensor, y_pred: Tensor):

        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.byte()

        tp = np.zeros_like(self.thresholds, dtype=np.uint64)
        tn = np.zeros_like(self.thresholds, dtype=np.uint64)
        fp = np.zeros_like(self.thresholds, dtype=np.uint64)
        fn = np.zeros_like(self.thresholds, dtype=np.uint64)

        for i, value in enumerate(self.thresholds):
            y_pred_i = torch.gt(y_pred, value)

            tp[i] = torch.eq(y_true, y_pred_i).sum().cpu().item()
            tn[i] = torch.eq(y_true == 0, y_pred_i == 0).sum().cpu().item()
            fp[i] = torch.eq(y_true == 0, y_pred_i == 1).sum().cpu().item()
            fn[i] = torch.eq(y_true == 1, y_pred_i == 0).sum().cpu().item()

        return tp, tn, fp, fn
