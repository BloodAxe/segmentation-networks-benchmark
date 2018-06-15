from torch import nn
import torch
from torch.nn import functional as F

class _Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class _DoubleConvModule(nn.Module):
    def __init__(self, in_: int, out: int, dropout_val, batch_norm):
        super().__init__()
        self.l1 = _Conv3BN(in_, out, batch_norm)
        self.l2 = _Conv3BN(out, out, batch_norm)
        self.dropout = nn.Dropout2d(p=dropout_val)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ZF_UNET(nn.Module):
    def __init__(self, dropout_val=0.2, batch_norm=True, input_channels=3, num_classes=1, filters=32):
        super(ZF_UNET, self).__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2)

        self.conv_224 = _DoubleConvModule(input_channels, filters, dropout_val, batch_norm)
        self.conv_112 = _DoubleConvModule(filters, 2 * filters, dropout_val, batch_norm)
        self.conv_56 = _DoubleConvModule(2 * filters, 4 * filters, dropout_val, batch_norm)
        self.conv_28 = _DoubleConvModule(4 * filters, 8 * filters, dropout_val, batch_norm)
        self.conv_14 = _DoubleConvModule(8 * filters, 16 * filters, dropout_val, batch_norm)

        self.conv_7 = _DoubleConvModule(16 * filters, 32 * filters, dropout_val, batch_norm)

        self.up_conv_14 = _DoubleConvModule(32 * filters + 16 * filters, 16 * filters, dropout_val, batch_norm)
        self.up_conv_28 = _DoubleConvModule(16 * filters + 8 * filters, 8 * filters, dropout_val, batch_norm)
        self.up_conv_56 = _DoubleConvModule(8 * filters + 4 * filters, 4 * filters, dropout_val, batch_norm)
        self.up_conv_112 = _DoubleConvModule(4 * filters + 2 * filters, 2 * filters, dropout_val, batch_norm)
        self.up_conv_224 = _DoubleConvModule(2 * filters + filters, filters, dropout_val, batch_norm)

        self.conv_final = nn.Conv2d(filters, num_classes, 1)

    def forward(self, x):
        conv_224 = self.conv_224(x)
        pool_112 = self.pool(conv_224)

        conv_112 = self.conv_112(pool_112)
        pool_56 = self.pool(conv_112)

        conv_56 = self.conv_56(pool_56)
        pool_28 = self.pool(conv_56)

        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool(conv_28)

        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool(conv_14)

        conv_7 = self.conv_7(pool_7)

        up_14 = torch.cat([self.unpool(conv_7), conv_14], dim=1)
        up_conv_14 = self.up_conv_14(up_14)

        up_28 = torch.cat([self.unpool(up_conv_14), conv_28], dim=1)
        up_conv_28 = self.up_conv_28(up_28)

        up_56 = torch.cat([self.unpool(up_conv_28), conv_56], dim=1)
        up_conv_56 = self.up_conv_56(up_56)

        up_112 = torch.cat([self.unpool(up_conv_56), conv_112], dim=1)
        up_conv_112 = self.up_conv_112(up_112)

        up_224 = torch.cat([self.unpool(up_conv_112), conv_224],dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        out = self.conv_final(up_conv_224)

        return out
