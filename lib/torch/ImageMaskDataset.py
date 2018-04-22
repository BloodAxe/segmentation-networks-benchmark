import torch
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    def __init__(self, image_filenames, target_filenames, image_loader, target_loader, transform=None):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        self.image_filenames = image_filenames
        self.target_filenames = target_filenames
        self.image_loader=image_loader
        self.target_loader=target_loader
        self.transform=transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        i = self.image_loader(self.image_filenames[index])
        t = self.target_loader(self.target_filenames[index])

        if self.transform is not None:
            i,t = self.transform(i,t)

        return i,t
