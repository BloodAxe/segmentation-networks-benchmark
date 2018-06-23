import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib import augmentations as aug
from lib.common import find_in_dir, TiledImagesDataset, read_rgb, read_mask, ImageMaskDataset
from lib.tiles import ImageSlicer


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


def INRIASliced(dataset_dir, grayscale):
    x = sorted(find_in_dir(os.path.join(dataset_dir, 'images')))
    y = sorted(find_in_dir(os.path.join(dataset_dir, 'gt')))
    image_id = [os.path.basename(fname).split('_')[0] for fname in x]

    unique_image_id = np.unique(image_id)
    location = [basename[:6] for basename in unique_image_id]  # Geocode is first 6 characters
    train_id, test_id = train_test_split(unique_image_id, random_state=1234, test_size=0.1, stratify=location)

    xy_train = [(image_fname, mask_fname) for image_fname, mask_fname, image_id in zip(x, y, image_id) if image_id in train_id]
    xy_test = [(image_fname, mask_fname) for image_fname, mask_fname, image_id in zip(x, y, image_id) if image_id in test_id]

    x_train, y_train = zip(*xy_train)
    x_test, y_test = zip(*xy_test)

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

    train = ImageMaskDataset(x_train, y_train, image_loader=read_rgb, target_loader=read_mask, transform=train_transform, load_in_ram=False)
    test = ImageMaskDataset(x_test, y_test, image_loader=read_rgb, target_loader=read_mask, transform=test_transform, load_in_ram=False)

    num_classes = 1
    return train, test, num_classes


def cut_dataset_in_patches(data_dir, output_dir, patch_size):
    x = sorted(find_in_dir(os.path.join(data_dir, 'images')))
    y = sorted(find_in_dir(os.path.join(data_dir, 'gt')))

    out_img = os.path.join(output_dir, 'images')
    out_msk = os.path.join(output_dir, 'gt')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    slicer = ImageSlicer((5000, 5000), patch_size, patch_size // 2)

    for image_fname, mask_fname in tqdm(zip(x, y), total=len(x)):
        image = read_rgb(image_fname)
        mask = read_mask(mask_fname)

        basename = os.path.basename(image_fname)
        basename = os.path.splitext(basename)[0]

        for index, patch in enumerate(slicer.split(image)):
            cv2.imwrite(os.path.join(out_img, '%s_%d.tif' % (basename, index)), patch)

        for index, patch in enumerate(slicer.split(mask)):
            cv2.imwrite(os.path.join(out_msk, '%s_%d.tif' % (basename, index)), patch)


if __name__ == '__main__':
    cut_dataset_in_patches('d:/datasets/inria/train', 'd:/datasets/inria-train-1024', 1024)
    # dataset_dir = 'e:/datasets/inria'
    # train = sorted(find_in_dir(os.path.join(dataset_dir, 'train', 'images')))
    # test = sorted(find_in_dir(os.path.join(dataset_dir, 'test', 'images')))
    # print('train', compute_mean_std(train))
    # print('test', compute_mean_std(test))
    # print('both', compute_mean_std(train + test))
