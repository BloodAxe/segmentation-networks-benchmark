from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import glob
from itertools import islice
import functools
from pathlib import Path
from pprint import pprint
import random
import shutil

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize, Compose
import tqdm
from torchvision.utils import make_grid

cuda_is_available = torch.cuda.is_available()


def maybe_cuda(x):
    return x.cuda() if cuda_is_available else x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_float_tensor(img: np.ndarray):
    # .copy() because RuntimeError: some of the strides of a given numpy array are negative.
    #  This is currently not supported, but will be added in future releases.
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
    return tensor

def show_landmarks_batch(data):
    x, y = data

    grid_x = make_grid(x, normalize=True, scale_each=True)
    grid_y = make_grid(y, normalize=True, scale_each=True)
    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(grid_x.numpy().transpose((1, 2, 0)))
    ax2.imshow(grid_y.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')
    plt.show()