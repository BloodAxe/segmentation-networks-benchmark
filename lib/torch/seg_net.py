import cv2
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from lib.torch.common import to_float_tensor, show_landmarks_batch, maybe_cuda


def _squash(p: torch.Tensor):
    p_sqr = p ** 2
    p_norm_sq = torch.sum(p_sqr, dim=1, keepdim=True)
    p_norm = torch.sqrt(p_norm_sq + 1e-9)
    v = p_norm_sq * p / ((1. + p_norm_sq) * p_norm)
    return v


def _compute_vector_length(p: torch.Tensor):
    p_sqr = p ** 2
    p_sum_sq = torch.sum(p_sqr, dim=1, keepdim=True)
    return torch.sqrt(p_sum_sq + 1e-9)


class RoutingSoftmax(nn.Module):
    def __init__(self, kernel_size, t):
        super(RoutingSoftmax, self).__init__()
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self.padding = nn.ZeroPad2d((1, 2, 1, 2))
        else:
            self.padding = nn.ZeroPad2d(kernel_size // 2)

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0)
        self.one_kernel = torch.ones([1, t, kernel_size, kernel_size])

        self.add_module('padding', self.padding)
        self.add_module('maxpool', self.maxpool)
        # self.register_buffer('one_kernel', self.one_kernel)

    def forward(self, b_t):
        b_t_pad = self.padding(b_t)
        b_t_max = self.maxpool(b_t_pad)

        if b_t.device.type == 'cuda':
            one_kernel = maybe_cuda(self.one_kernel)
        else:
            one_kernel = self.one_kernel

        b_t_max, _ = torch.max(b_t_max, dim=1, keepdim=True)
        c_t = torch.exp(b_t - b_t_max)  # [N, t_1, H_1, W_1]
        c_t_pad = self.padding(c_t)
        sum_c_t = F.conv2d(c_t_pad, one_kernel, stride=1, padding=0)
        r_t = c_t / (sum_c_t + 1e-9)  # [N, t_1, H_1, W_1]
        return r_t


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
        self.routing = routing
        self.routing_softmax = RoutingSoftmax(kernel_size, num_capsules)

        if mode == 'conv':
            pad = kernel_size // 2
            self.convOp = nn.Conv2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride, padding=pad)
        elif mode == 'deconv':
            self.convOp = nn.ConvTranspose2d(input_channels, capsule_dims * num_capsules, kernel_size=kernel_size, stride=stride, padding=1)

        self.add_module('convOp', self.convOp)
        self.add_module('routing_softmax', self.routing_softmax)

    def forward(self, x: torch.autograd.Variable):
        """

        :param x: [N, t_in, z_in, H, W]
        :return: Tensor of [N, t_out, z_out, H, W], where t is number of capsules, z is capsule dimension
        """

        t_out, z_out = self.num_capsules, self.out_capsule_dims

        N = x.size(0)
        t_in = x.size(1)
        z_in = x.size(2)
        H_in = x.size(3)
        W_in = x.size(4)

        u_t_list = [torch.squeeze(u_t, dim=1) for u_t in torch.split(x, 1, dim=1)]  # Splits into t_in chunks
        u_hat_t_list = []

        for u_t in u_t_list:
            # u_t: [N, z_0, H_0, W_0]
            u_hat_t = self.convOp(u_t)
            assert u_hat_t.size(1) == z_out * t_out
            # print(self.mode, u_t.size(), u_hat_t.size())

            H_1 = u_hat_t.size(2)
            W_1 = u_hat_t.size(3)

            # u_hat_t: [N, t_out, z_out, H_1, W_1]
            u_hat_t = u_hat_t.view([N, t_out, z_out, H_1, W_1])
            u_hat_t_list.append(u_hat_t)

        b = torch.zeros([N, t_out, t_in, H_1, W_1])
        if x.device.type == 'cuda':
            b = maybe_cuda(b)

        b_t_list = [torch.squeeze(b_t, dim=2) for b_t in torch.split(b, 1, dim=2)]  # Splits into t_in chunks
        u_hat_t_list_sg = [u_hat_t.detach() for u_hat_t in u_hat_t_list]

        for d in range(self.routing):
            if d < self.routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax
                r_t = self.routing_softmax(b_t)
                r_t = torch.unsqueeze(r_t, dim=2)  # [N, t_1, 1, H_1, W_1]
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
                    # print('b_t', b_t.size())
                    # print('u_hat_t', u_hat_t.size())
                    # print('v', v.size())

                    ss = u_hat_t * v
                    sss = torch.sum(ss, dim=2, keepdim=False)
                    b_t_list_.append(b_t + sss)
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

        self.reconstruction = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, self.input_channels, 1))

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

        x = torch.squeeze(x, dim=1)

        # 1. compute length of vector
        v_lens = _compute_vector_length(x)

        # 2. Get masked reconstruction
        recons = self.reconstruction(x)

        # if self.num_classes > 1:
        #     x = F.log_softmax(x, dim=1)

        return v_lens, recons


class SegCapsLoss:
    """
    Loss defined as class_loss + reconstruction_loss
    """

    def __init__(self, reconstruction_weight=0.0005):
        self.reconstruction_weight = reconstruction_weight

    def _reconstruction_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def _class_loss(self, outputs, targets):
        zero_input = torch.zeros_like(outputs)
        a = torch.max(zero_input, 0.9 - outputs)
        b = torch.max(zero_input, outputs - 0.1)
        x = 0.5 * targets * (a ** 2) + 0.5 * (1 - targets) * (b ** 2)
        return x.mean()

    def __call__(self, outputs, targets):
        v_lens, recons = outputs
        v_lens_true, recons_true = targets
        loss1 = self._class_loss(v_lens, v_lens_true)
        loss2 = self._reconstruction_loss(recons, recons_true) * self.reconstruction_weight
        return loss1 + loss2


def gen_random_image(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size, 1), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0 + 1, 255)
    light_color1 = random.randint(dark_color1 + 1, 255)
    light_color2 = random.randint(dark_color2 + 1, 255)
    center_0 = random.randint(0, size)
    center_1 = random.randint(0, size)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(size):
        for j in range(size):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, (mask > 0).astype(np.uint8)


class SyntheticShapes(Dataset):

    def __init__(self, size):
        self.size = size

    def __getitem__(self, index):
        img, mask = gen_random_image(self.size)
        return to_float_tensor(img), to_float_tensor(mask)

    def __len__(self):
        return 1024


if __name__ == "__main__":

    use_cuda = False

    trainloader = DataLoader(SyntheticShapes(512), batch_size=3, pin_memory=True)
    test_x, test_y = next(iter(trainloader))

    show_landmarks_batch((test_x, test_y))

    model = SegCaps(num_classes=1, input_channels=3)
    if use_cuda:
        model = model.cuda()

    criterion = SegCapsLoss()

    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for i, (x, y) in enumerate(trainloader):

            if use_cuda:
                x, y = x.cuda(), y.cuda()

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, (y, x))

            bs = x.size(0)
            (bs * loss).backward()

            optim.step()
            print(epoch, i, loss)

        model.test()
        y, rec = model(test_x)
        show_landmarks_batch((rec, y))
