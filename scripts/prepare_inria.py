import os
from os.path import splitext

import cv2
import numpy as np
from tqdm import tqdm

from lib.tiles import ImageSlicer
from torch_train import find_in_dir


def read_rgb(fname):
    x = cv2.imread(fname, cv2.IMREAD_COLOR)
    return x


def read_gray(fname):
    x = np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=-1)
    return x


def main():
    dataset_dir = 'e:\\datasets\\inria\\train\\'
    output_dir = 'e:\\datasets\\inria\\train_512'

    os.makedirs(os.path.join(output_dir,'images'),exist_ok=True)
    os.makedirs(os.path.join(output_dir,'gt'),exist_ok=True)
    images = find_in_dir(os.path.join(dataset_dir, 'images'))
    targets = find_in_dir(os.path.join(dataset_dir, 'gt'))

    for x, y in tqdm(zip(images, targets),total=len(images)):

        image_name = splitext(os.path.basename(x))[0]
        mask_name = splitext(os.path.basename(y))[0]

        x = read_rgb(x)
        y = read_gray(y)

        slicer = ImageSlicer(x.shape, 512, 256)
        xs = slicer.split(x)
        ys = slicer.split(y)

        for i, patch in enumerate(xs):
            cv2.imwrite(os.path.join(output_dir, 'images', '%s_%d.tif' % (image_name, i)), patch)

        for i, patch in enumerate(ys):
            cv2.imwrite(os.path.join(output_dir, 'gt', '%s_%d.tif' % (mask_name, i)), patch)


if __name__=='__main__':
    main()