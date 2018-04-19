import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-7
        return 1 - 2 * intersection / union


class JaccardScore(nn.Module):
    def __init__(self):
        super(JaccardScore, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-7
        return intersection / (union - intersection)


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.score = JaccardScore()

    def forward(self, output, target):
        return 1 - self.score(output, target)


class SmoothJaccardLoss(nn.Module):
    def __init__(self, smooth=100):
        super(SmoothJaccardLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.smooth = smooth

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-7

        jac = (intersection + self.smooth) / (union - intersection + self.smooth)
        return (1 - jac) * self.smooth


class BCEWithLogitsLossAndJaccard:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jac_loss = JaccardLoss()

        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss1 = self.nll_loss(outputs, targets)
        loss2 = self.jac_loss(outputs, targets)
        return (loss1 + loss2) / (1 + self.jaccard_weight)


class BCEWithLogitsLossAndSmoothJaccard:
    """
    Loss defined as BCE - SmoothJaccardLoss
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jac_loss = JaccardLoss()

        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss1 = self.nll_loss(outputs, targets)
        loss2 = self.jac_loss(outputs, targets)
        return (loss1 + loss2) / (1 + self.jaccard_weight)
