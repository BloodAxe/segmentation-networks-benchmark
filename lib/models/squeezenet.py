import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class DFire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(DFire, self).__init__()
        self.inplanes = inplanes

        self.expand1x1 = nn.Conv2d(inplanes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = nn.Conv2d(inplanes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ELU(inplace=True)

        self.squeeze = nn.Conv2d(expand3x3_planes + expand1x1_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        x = self.squeeze_activation(self.squeeze(x))
        return x

class SharpMaskBypass(nn.Module):
    def __init__(self, enc_features, dec_features, num_classes):
        super(SharpMaskBypass, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(enc_features, 32, kernel_size=3, padding=1), nn.ELU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32 + dec_features, num_classes, kernel_size=3, padding=1), nn.ELU(inplace=True))
        # TODO: Init weights stddev = 0.0001

    def forward(self, from_enc, from_dec):
        x = self.conv1(from_enc)
        x = torch.cat((x, from_dec), dim=1)
        x = self.conv2(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(SqueezeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fire4 = Fire(128, 48, 128, 128)
        self.fire5 = Fire(256, 48, 128, 128)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)

        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        self.conv10 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1), nn.ELU(inplace=True))
        self.dconv10 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1), nn.ELU(inplace=True))

        # Decoder
        self.dfire9 = DFire(512, 512, 256, 256)
        self.dfire8 = DFire(512, 384, 256, 256)
        self.dfire7 = DFire(384, 384, 192, 192)
        self.dfire6 = DFire(384, 256, 192, 192)
        self.dfire5 = DFire(256, 256, 128, 128)
        self.dfire4 = DFire(256, 128, 128, 128)
        self.dfire3 = DFire(128, 128, 64, 64)
        self.dfire2 = DFire(128, 96, 48, 48)

        self.dconv1 = nn.Conv2d(96, num_classes, kernel_size=1)

        #
        # self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        #
        # self.sharpmask3 = SharpMaskBypass(256, num_classes, num_classes)
        #
        # self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        #
        # self.sharpmask2 = SharpMaskBypass(128, num_classes, num_classes)
        #
        # self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        #
        # self.sharpmask1 = SharpMaskBypass(64, num_classes, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        fire2 = self.fire2(pool1)
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(fire3)
        pool4 = self.pool3(fire4)

        fire5 = self.fire5(pool4)
        fire6 = self.fire6(fire5)
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire7)

        pool8 = self.pool5(fire8)

        fire9 = self.fire9(pool8)
        center = self.dconv10(self.conv10(fire9))
        dfire9 = self.dfire9(center)

        dfire9 = F.upsample(dfire9, scale_factor=2, mode='nearest')
        dfire8 = self.dfire8(dfire9 + fire8)
        dfire7 = self.dfire7(dfire8)
        dfire6 = self.dfire6(dfire7)
        dfire5 = self.dfire5(dfire6)

        dfire5 = F.upsample(dfire5, scale_factor=2, mode='nearest')
        dfire4 = self.dfire4(dfire5 + fire4)
        dfire3 = self.dfire3(dfire4)
        dfire2 = self.dfire2(dfire3)

        dfire2 = F.upsample(dfire2, scale_factor=2, mode='nearest')
        dconv1 = self.dconv1(dfire2 + conv1)

        return dconv1

        # drop9 = self.drop9(fire9)
        # score_fr = self.score_fr(drop9)
        #
        # upscore2 = self.upscore2(score_fr)
        # sharpmask3 = self.sharpmask3(fire5, upscore2)
        #
        # upscore4 = self.upscore4(sharpmask3)
        # sharpmask2 = self.sharpmask2(fire3, upscore4)
        #
        # upscore8 = self.upscore8(sharpmask2)
        # sharpmask1 = self.sharpmask1(conv1, upscore8)
        #
        # return sharpmask1


if __name__ == '__main__':
    arch = SqueezeNet(in_channels=3, num_classes=1)

    x = torch.rand((4,3,512,512))
    y = arch(x)