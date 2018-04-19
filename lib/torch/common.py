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

cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x
