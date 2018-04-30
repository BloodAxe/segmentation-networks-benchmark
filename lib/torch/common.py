import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from lib.augmentations import ToTensors

cuda_is_available = torch.cuda.is_available()


def maybe_cuda(x):
    return x.cuda() if cuda_is_available else x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_landmarks_batch(data):
    x, y = data

    grid_x = make_grid(x, normalize=True, scale_each=True)
    grid_y = make_grid(y, normalize=True, scale_each=True)
    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(grid_x.numpy().transpose((1, 2, 0)))
    ax2.imshow(grid_y.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')
    plt.show()


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in os.listdir(dirname)]


def read_rgb(fname):
    x = cv2.imread(fname, cv2.IMREAD_COLOR)
    return x


def read_gray(fname):
    x = np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=-1)
    return x


def normalize_image(x: np.ndarray):
    x = x.astype(np.float32, copy=True)
    x /= 127.5
    x -= 1.
    return x


class RawDataset(Dataset):
    def __init__(self, images, masks, transform=ToTensors()):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, index):
        i, m = self.images[index], self.masks[index]
        return self.transform(i, m)

    def __len__(self):
        return len(self.images)


class ImageMaskDataset(Dataset):
    def __init__(self, image_filenames, target_filenames, image_loader, target_loader, transform=None, load_in_ram=False):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        if load_in_ram:
            self.image_filenames = [image_loader(fname) for fname in image_filenames]
            self.target_filenames = [target_loader(fname) for fname in target_filenames]
            self.image_loader = lambda x: x
            self.target_loader = lambda x: x
        else:
            self.image_filenames = image_filenames
            self.target_filenames = target_filenames
            self.image_loader = image_loader
            self.target_loader = target_loader

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        i = self.image_loader(self.image_filenames[index])
        t = self.target_loader(self.target_filenames[index])

        if self.transform is not None:
            i, t = self.transform(i, t)

        return i, t
