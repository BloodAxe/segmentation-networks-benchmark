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

    def __init__(self, thresholds=127):

        self.thresholds = np.arange(0., 1., 1. / thresholds, dtype=np.float32)
        n_thresholds = len(self.thresholds)

        self.tp = np.zeros(n_thresholds, dtype=np.uint64)
        self.tn = np.zeros(n_thresholds, dtype=np.uint64)
        self.fp = np.zeros(n_thresholds, dtype=np.uint64)
        self.fn = np.zeros(n_thresholds, dtype=np.uint64)
        self.precision = np.zeros(n_thresholds, dtype=np.float32)
        self.recall = np.zeros(n_thresholds, dtype=np.float32)

    def reset(self):
        self.tp.fill(0)
        self.tn.fill(0)
        self.fp.fill(0)
        self.fn.fill(0)
        self.precision.fill(0)
        self.recall.fill(0)

    def update(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

        self.precision = self.tp.astype(np.float32) / (self.tp + self.fp)
        self.recall = self.tp.astype(np.float32) / (self.tp + self.fn)
