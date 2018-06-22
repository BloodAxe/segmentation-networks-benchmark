import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from lib import augmentations as aug
from lib.common import find_in_dir, TiledImagesDataset, read_rgb


def compute_mean_std(dataset):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """
    one_over_255 = float(1. / 255.)

    global_mean = np.zeros(3, dtype=np.float64)
    global_var = np.zeros(3, dtype=np.float64)

    n_items = len(dataset)

    for image_fname in dataset:
        x = read_rgb(image_fname) * one_over_255
        mean, stddev = cv2.meanStdDev(x)

        global_mean += np.squeeze(mean)
        global_var += np.squeeze(stddev) ** 2

    return global_mean / n_items, np.sqrt(global_var)

INRIA_MEAN = [0.40273115, 0.45046371, 0.42960134]
INRIA_STD = [3.15086464, 3.29831641, 3.63201004]

def INRIA(dataset_dir, grayscale, patch_size, keep_in_mem, small=False):
    x = sorted(find_in_dir(os.path.join(dataset_dir, 'images')))
    y = sorted(find_in_dir(os.path.join(dataset_dir, 'gt')))

    if small:
        x = x[:4]
        y = y[:4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.1)


    train_transform = aug.Sequential([
        aug.ImageOnly(aug.RandomGrayscale(1.0 if grayscale else 0.5)),
        aug.ImageOnly(aug.RandomBrightness()),
        aug.ImageOnly(aug.RandomContrast()),
        aug.VerticalFlip(),
        aug.HorizontalFlip(),
        aug.ShiftScaleRotate(rotate_limit=15),
        aug.ImageOnly(aug.NormalizeImage(mean=INRIA_MEAN, std=INRIA_STD)),
        aug.MaskOnly(aug.MakeBinary())
    ])

    test_transform = aug.Sequential([
        aug.ImageOnly(aug.NormalizeImage(mean=INRIA_MEAN, std=INRIA_STD)),
        aug.MaskOnly(aug.MakeBinary())
    ])

    train = TiledImagesDataset(x_train, y_train, patch_size, target_shape=(5000, 5000), transform=train_transform, keep_in_mem=keep_in_mem)
    test = TiledImagesDataset(x_test, y_test, patch_size, target_shape=(5000, 5000), transform=test_transform, keep_in_mem=keep_in_mem)
    num_classes = 1
    return train, test, num_classes


if __name__ == '__main__':
    dataset_dir = 'e:/datasets/inria'
    train = sorted(find_in_dir(os.path.join(dataset_dir, 'train', 'images')))
    test = sorted(find_in_dir(os.path.join(dataset_dir, 'test', 'images')))
    print('train', compute_mean_std(train))
    print('test', compute_mean_std(test))
    print('both', compute_mean_std(train+test))

