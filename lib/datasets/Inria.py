import os

from sklearn.model_selection import train_test_split

from lib import augmentations as aug
from lib.common import find_in_dir, TiledImagesDataset


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
        aug.ImageOnly(aug.NormalizeImage()),
        aug.MaskOnly(aug.MakeBinary())
    ])

    test_transform = aug.Sequential([
        aug.ImageOnly(aug.NormalizeImage()),
        aug.MaskOnly(aug.MakeBinary())
    ])

    train = TiledImagesDataset(x_train, y_train, patch_size, target_shape=(5000,5000), transform=train_transform, keep_in_mem=keep_in_mem)
    test = TiledImagesDataset(x_test, y_test, patch_size, target_shape=(5000,5000), transform=test_transform, keep_in_mem=keep_in_mem)
    num_classes = 1
    return train, test, num_classes
