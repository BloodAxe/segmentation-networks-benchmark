import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import lib.augmentations as aug


def gen_random_image(patch_size):
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0 + 1, 255)
    light_color1 = random.randint(dark_color1 + 1, 255)
    light_color2 = random.randint(dark_color2 + 1, 255)
    center_0 = random.randint(0, patch_size)
    center_1 = random.randint(0, patch_size)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 1, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(patch_size):
        for j in range(patch_size):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask


class ShapesDataset(Dataset):
    def __init__(self, steps, patch_size, transform=aug.ImageOnly(aug.NormalizeImage())):
        self.transform = transform
        self.patch_size = patch_size
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, item):
        image, mask = gen_random_image(self.patch_size)
        image, mask = self.transform(image, mask)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float()
        mask = torch.from_numpy(np.expand_dims(mask, 0)).long()
        return image, mask


def SHAPES(patch_size):
    """
    https://github.com/ZFTurbo/ZF_UNET_patch_size_Pretrained_Model/blob/master/train_infinite_generator.py
    :param patch_size:
    :return:
    """
    return ShapesDataset(1024, patch_size), ShapesDataset(128, patch_size), 1
