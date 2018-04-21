import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class SegCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(SegCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
#
# def softmax(input, dim=1):
#     transposed_input = input.transpose(dim, len(input.size()) - 1)
#     softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
#     return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
#
#
# class CapsuleLayer(nn.Module):
#     def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
#                  num_iterations=NUM_ROUTING_ITERATIONS):
#         super(CapsuleLayer, self).__init__()
#
#         self.num_route_nodes = num_route_nodes
#         self.num_iterations = num_iterations
#
#         self.num_capsules = num_capsules
#
#         if num_route_nodes != -1:
#             self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
#         else:
#             self.capsules = nn.ModuleList(
#                 [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
#                  range(num_capsules)])
#
#     def squash(self, tensor, dim=-1):
#         squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
#         scale = squared_norm / (1 + squared_norm)
#         return scale * tensor / torch.sqrt(squared_norm)
#
#     def forward(self, x):
#         if self.num_route_nodes != -1:
#             priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
#
#             logits = Variable(torch.zeros(*priors.size())).cuda()
#             for i in range(self.num_iterations):
#                 probs = softmax(logits, dim=2)
#                 outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
#
#                 if i != self.num_iterations - 1:
#                     delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
#                     logits = logits + delta_logits
#         else:
#             outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
#             outputs = torch.cat(outputs, dim=-1)
#             outputs = self.squash(outputs)
#
#         return outputs
#
#
# class CapsuleNet(nn.Module):
#     def __init__(self):
#         super(CapsuleNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
#         self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
#                                              kernel_size=9, stride=2)
#         self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
#                                            out_channels=16)
#
#         self.decoder = nn.Sequential(
#             nn.Linear(16 * NUM_CLASSES, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 784),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, y=None):
#         x = F.relu(self.conv1(x), inplace=True)
#         x = self.primary_capsules(x)
#         x = self.digit_capsules(x).squeeze().transpose(0, 1)
#
#         classes = (x ** 2).sum(dim=-1) ** 0.5
#         classes = F.softmax(classes, dim=-1)
#
#         if y is None:
#             # In all batches, get the most active capsule.
#             _, max_length_indices = classes.max(dim=1)
#             y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
#
#         reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
#
#         return classes, reconstructions
#
#
# class CapsuleLoss(nn.Module):
#     def __init__(self):
#         super(CapsuleLoss, self).__init__()
#         self.reconstruction_loss = nn.MSELoss(size_average=False)
#
#     def forward(self, images, labels, classes, reconstructions):
#         left = F.relu(0.9 - classes, inplace=True) ** 2
#         right = F.relu(classes - 0.1, inplace=True) ** 2
#
#         margin_loss = labels * left + 0.5 * (1. - labels) * right
#         margin_loss = margin_loss.sum()
#
#         assert torch.numel(images) == torch.numel(reconstructions)
#         images = images.view(reconstructions.size()[0], -1)
#         reconstruction_loss = self.reconstruction_loss(reconstructions, images)
#
#         return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
