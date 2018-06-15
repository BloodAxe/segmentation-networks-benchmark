import torch
from torch import nn, Tensor
from torch.nn import functional as F, BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        prediction = F.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-7
        return 1 - 2 * intersection / union


class JaccardLoss(_Loss):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, output, target):
        output = F.sigmoid(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)

        jac = intersection / (union - intersection + 1e-7)
        return 1 - jac


class SmoothJaccardLoss(_Loss):
    def __init__(self, smooth=100):
        super(SmoothJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        output = F.sigmoid(output)
        target = target.float()
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)

        jac = (intersection + self.smooth) / (union - intersection + self.smooth)
        return 1 - jac


class BCEWithSigmoidLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super().__init__(size_average=size_average, reduce=reduce)

    def forward(self, outputs, targets):
        outputs = F.logsigmoid(outputs)
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(outputs, targets, size_average=self.size_average, reduce=self.reduce)


class BCEWithLogitsLossAndSmoothJaccard(_Loss):
    """
    Loss defined as BCE + SmoothJaccardLoss
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, bce_weight=1, jaccard_weight=0.5):
        super(BCEWithLogitsLossAndSmoothJaccard, self).__init__()
        self.bce_loss = BCEWithSigmoidLoss()
        self.jac_loss = SmoothJaccardLoss()

        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight

    def forward(self, outputs, targets):
        loss1 = self.bce_loss(outputs, targets) * self.bce_weight
        loss2 = self.jac_loss(outputs, targets) * self.jaccard_weight
        return (loss1 + loss2) / (self.bce_weight + self.jaccard_weight)


class FocalLossBinary(_Loss):
    """Focal loss puts more weight on more complicated examples.
    https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/losses.py
    output is log_softmax
    """

    def __init__(self, gamma=2, size_average=True, reduce=True):
        super(FocalLossBinary, self).__init__(size_average=size_average, reduce=reduce)
        self.gamma = gamma

    def forward(self, outputs: Tensor, targets: Tensor):

        outputs = F.logsigmoid(outputs)
        logpt = -F.binary_cross_entropy_with_logits(outputs, targets.float(), reduce=False)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
