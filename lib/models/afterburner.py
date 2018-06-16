import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.unet import UNet


class Afterburner(nn.Module):
    def __init__(self,n_channels=1):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, n_classes=1)

    def forward(self, x):
        return self.unet(x)

