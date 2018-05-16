import torch
from torch import nn
from torch.nn import functional as F

# Port of https://github.com/lalonderodney/SegCaps/blob/master/capsule_layers.py

class Length(nn.Module):
    def __init__(self, dim, num_capsules_dim):
        super(Length, self).__init__()
        self.dim = dim
        self.num_capsules_dim = num_capsules_dim

    def forward(self, x:torch.Tensor):
        assert x.size(self.num_capsules_dim) == 1
        x = torch.squeeze(x, dim=self.num_capsules_dim)
        x = torch.norm(x,dim=self.dim,keepdim=True)
        return x

class Decoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Decoder, self).__init__()
        self.convlist = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, num_classes, 1))

        if self.num_classes > 1:
            self.activation = nn.LogSoftmax()
        else:
            self.activation = nn.Sigmoid()

        self.num_classes = num_classes

    def forward(self, x):
        x = self.convlist(x)
        x = self.activation(x)
        return x

class ConvCapsuleLayer(nn.Module):

class DeconvCapsuleLayer(nn.Module):



class CapsNetR3(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CapsNetR3, self).__init__()

        self.conv1 = nn.Sequential([[
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_channels, 16, 5, 1),
            nn.ReLU()
        ]])

        self.primarycaps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same', routings=1, name='primarycaps')

    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x) # (N, C, H, W)
        x = x.unsqueeze(dim=1) # Reshape layer to be 1 capsule x [filters] atoms (N, 1, C, H, W)

        self.primary_caps(x)
