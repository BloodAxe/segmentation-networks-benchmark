import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import numpy as np


def _squash(p: torch.FloatTensor):
    p_norm_sq = torch.sum(p ** 2, dim=2, keepdim=True)
    p_norm = torch.sqrt(p_norm_sq + 1e-9)
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v


def _compute_vector_length(x):
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1e-9)


class CapsuleLayer(nn.Module):
    def __init__(self, mode, input_channels, kernel_size, stride, num_capsules, capsule_dims, routing):
        """

        :param input_channels: The input dimension of each capsule
        :param capsule_dims: The output dimension of each capsule
        :param kernel_size: Kernel size of (de)convolution and routing
        :param stride: Stride size of (de)convotluion
        :param num_capsules: The number of capsules
        :param routing: The number of routing
        """
        super(CapsuleLayer, self).__init__()

        self.input_channels = input_channels
        self.out_capsule_dims = capsule_dims
        self.kernel_size = kernel_size
        self.num_capsules = num_capsules
        self.mode = mode

        if mode == 'conv':
            pad = kernel_size // 2
            self.convOp = nn.Conv2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride, padding=pad)

            # self.convOp = nn.Sequential(nn.ReplicationPad2d(pad),
            #                             nn.Conv2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride))

            self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
        else:
            pad = kernel_size // 2
            self.convOp = nn.ConvTranspose2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride, padding=1)
            # self.convOp = nn.Sequential(nn.ReplicationPad2d(pad),
            #                             nn.ConvTranspose2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride))
            self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1,padding=1)

        self.one_kernel = Variable(torch.ones([1, 1, kernel_size, kernel_size]))
        self.routing = routing

    def forward(self, x: torch.autograd.Variable):
        # X is [N, num_capsules, in_capsule_dims, H, W]

        t_1, z_1 = self.num_capsules, self.out_capsule_dims

        N = x.size(0)
        t_0 = x.size(1)
        z_0 = x.size(2)
        H_0 = x.size(3)
        W_0 = x.size(4)

        u_t_list = [torch.squeeze(u_t, 1) for u_t in torch.chunk(x, t_0, dim=1)]
        u_hat_t_list = []
        for u_t in u_t_list:
            # u_t: [N, z_0, H_0, W_0]
            if self.mode == 'conv':
                u_hat_t = self.convOp(u_t)
            else:
                out_size = [N, t_1 * z_1, H_0 * 2, W_0 * 2]
                u_hat_t = self.convOp(u_t, output_size=out_size)
                print(self.mode, x.size(), out_size, u_hat_t.size())

            H_1 = u_hat_t.size(2)
            W_1 = u_hat_t.size(3)
            u_hat_t = u_hat_t.view([N, t_1, z_1, H_1, W_1])
            u_hat_t_list.append(u_hat_t)

        b = Variable(torch.zeros([N, t_1, t_0, H_1, W_1]))
        b_t_list = [torch.squeeze(b_t, dim=2) for b_t in torch.chunk(b, t_0, dim=2)]
        u_hat_t_list_sg = [u_hat_t.detach() for u_hat_t in u_hat_t_list]

        for d in range(self.routing):
            if d < self.routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax

                b_t_max = self.maxpool(b_t)
                r_t = F.softmax(b_t_max, dim=1)

                # sum_c_t = c_t.sum F.conv2d(c_t, self.one_kernel, stride=1, padding=k // 2)  # [... , 1]
                # r_t = c_t / sum_c_t  # [N, H_1, W_1, t_1]

                r_t = torch.unsqueeze(r_t, dim=2)  # [N, 1, t_1, H_1, W_1]
                print('r_t', r_t.size())
                print('u_hat_t', u_hat_t.size())
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, z_1, t_1, H_1, W_1]

            # p = tf.add_n(r_t_mul_u_hat_t_list)  # [N, z_1, t_1, H_1, W_1]
            p = torch.zeros_like(r_t_mul_u_hat_t_list[0])
            for x in r_t_mul_u_hat_t_list:
                p += x

            v = _squash(p)

            if d < self.routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    # b_t     : [N, t_1, H_1, W_1]
                    # u_hat_t : [N, z_1, t_1, H_1, W_1]
                    # v       : [N, z_1, t_1, H_1, W_1]
                    print('b_t', b_t.size())
                    print('u_hat_t', u_hat_t.size())
                    print('v', v.size())

                    b_t_list_.append(b_t + torch.sum(u_hat_t * v, dim=2, keepdim=False))
                b_t_list = b_t_list_

        return v


class SegCaps(nn.Module):
    """
    https://arxiv.org/pdf/1804.04241.pdf
    https://github.com/iwyoo/tf-SegCaps/blob/master/model.py
    """
    def __init__(self, num_classes=1, input_channels=3):
        super(SegCaps, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        self.firstconv = nn.Conv2d(input_channels, 16, 5, padding=2)
        self.enc1 = nn.Sequential(CapsuleLayer('conv', input_channels=16, capsule_dims=16, kernel_size=5, stride=2, num_capsules=2, routing=1),
                                  CapsuleLayer('conv', input_channels=16, capsule_dims=16, kernel_size=5, stride=1, num_capsules=4, routing=3))
        self.enc2 = nn.Sequential(CapsuleLayer('conv', input_channels=16, capsule_dims=32, kernel_size=5, stride=2, num_capsules=4, routing=3),
                                  CapsuleLayer('conv', input_channels=32, capsule_dims=32, kernel_size=5, stride=1, num_capsules=8, routing=3))
        self.enc3 = nn.Sequential(CapsuleLayer('conv', input_channels=32, capsule_dims=64, kernel_size=5, stride=2, num_capsules=8, routing=3),
                                  CapsuleLayer('conv', input_channels=64, capsule_dims=32, kernel_size=5, stride=1, num_capsules=8, routing=3))

        self.dec31 = CapsuleLayer('deconv', input_channels=32, capsule_dims=32, kernel_size=4, stride=2, num_capsules=8, routing=3)
        self.dec32 = CapsuleLayer('conv', input_channels=32, capsule_dims=32, kernel_size=5, stride=1, num_capsules=4, routing=3)

        self.dec21 = CapsuleLayer('deconv', input_channels=32, capsule_dims=16, kernel_size=4, stride=2, num_capsules=4, routing=3)
        self.dec22 = CapsuleLayer('conv', input_channels=16, capsule_dims=16, kernel_size=5, stride=1, num_capsules=4, routing=3)

        self.dec11 = CapsuleLayer('deconv', input_channels=16, capsule_dims=16, kernel_size=4, stride=2, num_capsules=2, routing=3)
        self.dec12 = CapsuleLayer('conv', input_channels=16, capsule_dims=16, kernel_size=1, stride=1, num_capsules=1, routing=3)

    def forward(self, x):
        x = self.firstconv(x)

        x = torch.unsqueeze(x, dim=1)  # [N, t=1, z, H, W]
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
        x = torch.cat([x, skip3], dim=1)
        x = self.dec32(x)

        # 1/2
        x = self.dec21(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec22(x)

        # 1
        x = self.dec11(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec12(x)

        x = x.squeeze(x, axis=1)

        # 1. compute length of vector
        v_lens = self.compute_vector_length(x)

        # 2. Get masked reconstruction
        x = self.conv2d(x, 64, 1)
        x = self.conv2d(x, 128, 1)
        recons = self.conv2d(x, self.input_channels, 1)

        # if self.num_classes > 1:
        #     x = F.log_softmax(x, dim=1)

        return v_lens, recons


x = np.random.rand(1, 3, 512, 512)
x = torch.from_numpy(x.astype(np.float32))

model = SegCaps(num_classes=1, input_channels=3)
model.eval()
y, r = model.forward(Variable(x))
