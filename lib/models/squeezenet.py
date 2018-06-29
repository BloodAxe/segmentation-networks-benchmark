import math
import torch
import torch.nn as nn
import torch.nn.init as init


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

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.fire2 = Fire(64, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fire4 = Fire(128, 128, 128, 128)
        self.fire5 = Fire(256, 128, 128, 128)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)

        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        self.drop9 = nn.Dropout2d(0.5)
        self.score_fr = nn.Conv2d(512, num_classes, kernel_size=1)

        # Decoder

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)

        self.sharpmask3 = SharpMaskBypass(256, num_classes, num_classes)

        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)

        self.sharpmask2 = SharpMaskBypass(128, num_classes, num_classes)

        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)

        self.sharpmask1 = SharpMaskBypass(64, num_classes, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        fire2 = self.fire2(pool1)
        fire3 = self.fire3(fire2)
        pool3 = self.pool3(fire3)

        fire4 = self.fire4(pool3)
        fire5 = self.fire5(fire4)
        pool5 = self.pool5(fire5)

        fire6 = self.fire6(pool5)
        fire7 = self.fire7(fire6)

        fire8 = self.fire8(fire7)
        fire9 = self.fire9(fire8)

        drop9 = self.drop9(fire9)
        score_fr = self.score_fr(drop9)

        upscore2 = self.upscore2(score_fr)
        sharpmask3 = self.sharpmask3(fire5, upscore2)

        upscore4 = self.upscore4(sharpmask3)
        sharpmask2 = self.sharpmask2(fire3, upscore4)

        upscore8 = self.upscore8(sharpmask2)
        sharpmask1 = self.sharpmask1(conv1, upscore8)

        return sharpmask1


if __name__ == '__main__':
    arch = SqueezeNet(in_channels=3, num_classes=1)

    x = torch.rand((4,3,512,512))
    y = arch(x)