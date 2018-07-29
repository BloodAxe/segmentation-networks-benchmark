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



class FocalLossMulti(_Loss):
    """Focal loss puts more weight on more complicated examples.
    https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/losses.py
    output is log_softmax
    """

    def __init__(self, gamma=2, size_average=True, reduce=True, ignore_index=-100, from_logits=False):
        super(FocalLossMulti, self).__init__(size_average=size_average, reduce=reduce)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, outputs: Tensor, targets: Tensor):

        if not self.from_logits:
            outputs = F.log_softmax(outputs, dim=1)

        logpt = -F.nll_loss(outputs, targets, ignore_index=self.ignore_index, reduce=False)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class JaccardLossMulti(_Loss):
    """
    Multiclass jaccard loss
    """

    def __init__(self, ignore_index=-100, from_logits=False, weight: Tensor = None, reduce=True):
        super(JaccardLossMulti, self).__init__(reduce=reduce)
        self.ignore_index = ignore_index
        self.from_logits = from_logits

        if weight is None:
            self.class_weights = None
        else:
            self.class_weights = weight / weight.sum()

    def forward(self, outputs: Tensor, targets: Tensor):
        """

        :param outputs: NxCxHxW
        :param targets: NxHxW
        :return: scalar
        """
        if self.from_logits:
            outputs = outputs.exp()
        else:
            outputs = F.softmax(outputs, dim=1)

        n_classes = outputs.size(1)
        mask = (targets != self.ignore_index)
        smooth = 100

        loss = torch.zeros(n_classes, dtype=torch.float).to(outputs.device)

        for cls_indx in range(0, outputs.size(1)):
            jaccard_target = (targets == cls_indx)
            jaccard_output = outputs[:, cls_indx]

            jaccard_target = torch.masked_select(jaccard_target, mask)
            jaccard_output = torch.masked_select(jaccard_output, mask)

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[cls_indx] = 0
            else:
                jaccard_target = jaccard_target.float()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                jac = (intersection + smooth) / (union - intersection + smooth)
                loss[cls_indx] = 1 - jac

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(outputs.device)

        if self.reduce:
            return loss.sum()

        return loss


class FocalAndJaccardLossMulti(_Loss):
    def __init__(self, jaccard_weight=1, class_weights=None, ignore_index=-1):
        super(FocalAndJaccardLossMulti, self).__init__()

        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights).float()
        else:
            nll_weight = None

        self.focal_loss = FocalLossMulti(ignore_index=ignore_index, from_logits=True)
        self.jaccard_loss = JaccardLossMulti(ignore_index=ignore_index, from_logits=True, weight=nll_weight)
        self.jaccard_weight = jaccard_weight

    def forward(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim=1)
        focal_loss = self.focal_loss(outputs, targets)
        jac_loss = self.jaccard_loss(outputs, targets)
        return (focal_loss + jac_loss) / (1 + self.jaccard_weight)


class NLLLAndJaccardLossMulti(_Loss):
    def __init__(self, jaccard_weight=1, class_weights=None, ignore_index=-1):
        super(NLLLAndJaccardLossMulti, self).__init__()

        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights).float()
        else:
            nll_weight = None

        self.nll_loss = NLLLoss(weight=nll_weight, ignore_index=ignore_index)
        self.jaccard_loss = JaccardLossMulti(ignore_index=ignore_index, from_logits=True, weight=nll_weight)
        self.jaccard_weight = jaccard_weight

    def forward(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim=1)
        nll_loss = self.nll_loss(outputs, targets)
        jac_loss = self.jaccard_loss(outputs, targets)
        return (nll_loss + jac_loss) / (1 + self.jaccard_weight)
