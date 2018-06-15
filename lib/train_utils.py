import glob
import os

import torch

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
