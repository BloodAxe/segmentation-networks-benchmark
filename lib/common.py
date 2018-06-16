import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.utils import make_grid

from lib.tiles import ImageSlicer

cuda_is_available = torch.cuda.is_available()


def maybe_cuda(x):
    return x.cuda() if cuda_is_available else x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

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


def read_mask(fname):
    x = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    return x


class InMemoryDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, index):
        i = self.images[index].copy()

        if self.masks is not None:
            m = self.masks[index].copy()
        else:
            m = None

        if self.transform is not None:
            i, m = self.transform(i, m)

        i = torch.from_numpy(np.moveaxis(i, -1, 0)).float()

        if self.masks is not None:
            m = torch.from_numpy(np.expand_dims(m, 0)).long()
            return i, m
        else:
            return i

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
        image = self.image_loader(self.image_filenames[index])
        mask = self.target_loader(self.target_filenames[index])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float()
        mask = torch.from_numpy(np.expand_dims(mask, 0)).long()
        return image, mask


class TiledImageDataset(Dataset):
    def __init__(self, image_fname, mask_fname, tile_size, tile_step=0, image_margin=0, transform=None, keep_in_mem=False):
        self.image_fname = image_fname
        self.mask_fname = mask_fname

        image = read_rgb(image_fname)
        mask = read_mask(mask_fname)
        self.image = image if keep_in_mem else None
        self.mask = mask if keep_in_mem else None

        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            raise ValueError()

        if tile_step <= 0:
            tile_step = tile_size//2

        self.slicer = ImageSlicer(image.shape, tile_size, tile_step, image_margin)
        self.transform = transform

    def __len__(self):
        return len(self.slicer.crops)

    def __getitem__(self, index):
        image = self.image if self.image is not None else read_rgb(self.image_fname)
        mask = self.mask if self.mask is not None else read_mask(self.mask_fname)

        image = self.slicer.cut_patch(image, index).copy()
        mask = self.slicer.cut_patch(mask, index).copy()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float()
        mask = torch.from_numpy(np.expand_dims(mask, 0)).long()
        return image, mask


class TiledImagesDataset(ConcatDataset):
    def __init__(self, image_filenames, target_filenames, tile_size, tile_step=0, image_margin=0, transform=None,keep_in_mem=False):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        datasets = [TiledImageDataset(image, mask, tile_size, tile_step, image_margin, transform,keep_in_mem=keep_in_mem) for image, mask in zip(image_filenames, target_filenames)]
        super().__init__(datasets)