import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CocoDetection

from lib.torch.common import find_in_dir, ImageMaskDataset, read_rgb, read_gray
from lib import augmentations as aug
from torchvision import transforms as trans

def COCO(dataset_dir, grayscale, patch_size):
    train_transform = trans.Compose([
        trans.RandomCrop(patch_size),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.RandomGrayscale(),
        trans.ToTensor(),
    ])

    test_transform = trans.Compose([
        trans.CenterCrop(patch_size),
        trans.transforms.ToTensor(),
    ])

    num_classes = 182
    return CocoDetection(root=os.path.join(dataset_dir, 'train2014'),
                         annFile=os.path.join(dataset_dir, 'annotations_trainval2014', 'instances_train2014.json'),
                         transform=train_transform), \
           CocoDetection(root=os.path.join(dataset_dir, 'val2014'),
                         annFile=os.path.join(dataset_dir, 'annotations_trainval2014', 'instances_val2014.json'),
                         transform=test_transform), \
           num_classes