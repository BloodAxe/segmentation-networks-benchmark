import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F



class CapsuleLayer(nn.Module):
    def __init__(self, k, s, t, z, routing):
        super(CapsuleLayer, self).__init__()

        self.capsules = nn.ModuleList(
            [nn.Conv2d(in_channels, z, kernel_size=kernel_size, stride=stride, padding=0) for _ in
             range(num_capsules)])

    def forward(self, x):
        pass


class DeconvCapsuleLayer(nn.Module):
    def __init__(self, k, s, t, z, routing):
        super(DeconvCapsuleLayer, self).__init__()

    def forward(self, x):
        pass


class SegCaps(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super(SegCaps, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        self.firstconv = nn.Conv2d(input_channels, 16, 5)
        self.enc1 = nn.Sequential(CapsuleLayer(k=5, s=2, t=2, z=16, routing=1), CapsuleLayer(k=5, s=1, t=4, z=16, routing=3))
        self.enc2 = nn.Sequential(CapsuleLayer(k=5, s=2, t=4, z=32, routing=3), CapsuleLayer(k=5, s=1, t=8, z=32, routing=3))
        self.enc3 = nn.Sequential(CapsuleLayer(k=5, s=2, t=8, z=64, routing=3), CapsuleLayer(k=5, s=1, t=8, z=32, routing=3))

        self.dec31 = DeconvCapsuleLayer(k=4, s=2, t=8, z=32, routing=3)
        self.dec32 = CapsuleLayer(k=5, s=1, t=4, z=32, routing=3)

        self.dec21 = DeconvCapsuleLayer( k=4, s=2, t=4, z=16, routing=3)
        self.dec22 = CapsuleLayer(k=5, s=1, t=4, z=16, routing=3)

        self.dec11 = DeconvCapsuleLayer(k=4, s=2, t=2, z=16, routing=3)
        self.dec12 = CapsuleLayer(k=1, s=1, t=1, z=16, routing=3)

    def forward(self, x: torch.FloatTensor):

        x = self.firstconv(x)

        x = x.expand(axis=3)  # [N, H, W, t=1, z]
        skip1 = x

        # 1/2
        x = self.enc1(x)
        skip2 = x

        # 1/4
        x = self.enc2(x)
        skip3 = x

        # 1/8
        x = self.enc3(x)

        # 1/4
        x = self.dec31(x)
        x = torch.cat([x, skip3], dim=3)
        x = self.dec32(x)

        # 1/2
        x = self.dec21(x)
        x = torch.cat([x, skip2], dim=3)
        x = self.dec22(x)

        # 1
        x = self.dec11(x)
        x = torch.cat([x, skip1], dim=3)
        x = self.dec12(x)

        x = x.squeeze(x, axis=3)

        # 1. compute length of vector
        v_lens = self.compute_vector_length(x)

        # 2. Get masked reconstruction
        x = self.conv2d(x, 64, 1)
        x = self.conv2d(x, 128, 1)
        recons = self.conv2d(x, self.input_channels, 1)

        if self.num_classes > 1:
            x = F.log_softmax(x, dim=1)

        return v_lens, recons