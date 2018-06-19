import glob
import os

import torch
from sklearn.metrics import confusion_matrix

from torch import nn
from torch.optim import Optimizer, SGD
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '%.3f' % self.avg


def find_optimal_lr(model: nn.Module, criterion, optimizer: Optimizer, dataloader):
    min_lr = 1e-8
    lrs = []
    lr = min_lr
    for i in range(30):
        lrs.append(lr)
        lr *= 2.

    lrs = np.array(lrs, dtype=np.float32)
    print(lrs)

    loss = np.zeros_like(lrs)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lrs[x])

    with torch.set_grad_enabled(True):
        model.train()
        dataiter = iter(dataloader)
        for i, lr in enumerate(tqdm(lrs, total=len(lrs))):
            scheduler.step()
            x, y = next(dataiter)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            y_pred = model(x)
            batch_loss = criterion(y_pred, y)

            batch_size = x.size(0)
            (batch_size * batch_loss).backward()

            optimizer.step()

            loss[i] = batch_loss.cpu().item()

    return lrs, loss


def auto_file(filename, where='.') -> str:
    """
    Helper function to find a unique filename in subdirectory without specifying fill path to it
    :param filename:
    :return:
    """
    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return filename

    files = list(glob.iglob(os.path.join(where, '**', filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError('Given file could not be found with recursive search:' + filename)

    if len(files) > 1:
        raise FileNotFoundError('More than one file matches given filename. Please specify it explicitly' + filename)

    return files[0]


class PRCurveMeter(object):

    def __init__(self, n_thresholds=127):
        self.n_thresholds = n_thresholds
        self.k = 2
        self.thresholds = np.arange(0., 1., 1. / n_thresholds, dtype=np.float32)
        self.tp = np.zeros(n_thresholds, dtype=np.uint64)
        self.tn = np.zeros(n_thresholds, dtype=np.uint64)
        self.fp = np.zeros(n_thresholds, dtype=np.uint64)
        self.fn = np.zeros(n_thresholds, dtype=np.uint64)

    def reset(self):
        self.tp.fill(0)
        self.tn.fill(0)
        self.fp.fill(0)
        self.fn.fill(0)

    def update(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred.detach()).cpu().numpy().reshape(-1)
        y_true = y_true.cpu().numpy().astype(np.int32).reshape(-1)

        for i, value in enumerate(self.thresholds):
            y_pred_i = (y_pred > value).astype(np.int32)

            # hack for bincounting 2 arrays together
            x = y_pred_i + self.k * y_true
            bincount_2d = np.bincount(x, minlength=self.k ** 2)
            assert bincount_2d.size == self.k ** 2
            conf = bincount_2d.reshape((self.k, self.k))

            self.tp[i] += conf[1, 1]
            self.tn[i] += conf[0, 0]
            self.fp[i] += conf[0, 1]
            self.fn[i] += conf[1, 0]

    def precision(self):
        return np.divide(self.tp, self.tp + self.fp)

    def recall(self):
        return np.divide(self.tp, self.tp + self.fn)
