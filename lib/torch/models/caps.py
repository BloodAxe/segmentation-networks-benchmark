import torch
from torch import nn, Tensor
from torch.nn import functional as F


# Port of https://github.com/lalonderodney/SegCaps/blob/master/capsule_layers.py

class Length(nn.Module):
    def __init__(self, dim, capsules_dim, seg=True):
        super(Length, self).__init__()
        self.dim = dim
        self.capsules_dim = capsules_dim

    def forward(self, x: torch.Tensor):
        assert x.size(self.capsules_dim) == 1
        x = torch.squeeze(x, dim=self.capsules_dim)
        x = torch.norm(x, dim=self.dim, keepdim=True)
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


class UpdateRouting(nn.Module):
    def __init__(self, n):
        super(UpdateRouting, self).__init__()
        self.routings = n

    def forward(self, votes:Tensor, logits:Tensor):

        activations = torch.zeros_like(votes)

        for i in self.routings:
            route = F.softmax(logits, dim=-1)
            preactivate_unrolled = route * votes.t()
            preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
            preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
            activation = _squash(preactivate)
            activations = activations.write(i, activation)
            act_3d = K.expand_dims(activation, 1)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
            tile_shape[1] = input_dim
            act_replicated = tf.tile(act_3d, tile_shape)
            distances = tf.reduce_sum(votes * act_replicated, axis=-1)
            logits += distances

        return activations, logits


class ConvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, num_capsule, num_atoms, stride, routings, padding: int):
        super(ConvCapsuleLayer, self).__init__()
        self.convOp = nn.Conv2d(num_atoms, num_capsule * num_atoms, kernel_size=kernel_size, stride=stride, padding=padding)
        self.routing = nn.Sequential([UpdateRouting() for i in range(routings)])

    def forward(self, x):
        activations = self.convOp(x)
        logits = torch.zeros_like(activations) + 0.1
        activations, logits = self.routing(activations, logits)
        return activations


class DeconvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, num_capsule, num_atoms, routings, upsamp_type, scaling, padding='same'):
        super(DeconvCapsuleLayer, self).__init__()
        self.routing = nn.Sequential([UpdateRouting() for i in range(routings)])

    def forward(self, x):
        pass


class CapsNetR3(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CapsNetR3, self).__init__()

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Sequential([[
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_channels, 16, 5, 1),
            nn.ReLU()
        ]])

        # Reshape layer to be 1 capsule x [filters] atoms
        # _, H, W, C = conv1.get_shape()
        # conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        self.primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, stride=2, padding=2,
                                             routings=1)

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, stride=1, padding=2,
                                             routings=3)

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, stride=2, padding=2,
                                             routings=3)

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, stride=1, padding=2,
                                             routings=3)

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, stride=2, padding=2,
                                             routings=3)

        # Layer 4: Convolutional Capsule
        self.conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, stride=1, padding=2,
                                             routings=3)

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                                 scaling=2, padding='same', routings=3)

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, stride=1,
                                               padding=2, routings=3)

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                                 scaling=2, padding='same', routings=3)

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, stride=1,
                                               padding=2, routings=3)

        # Layer 3 Up: Deconvolutional Capsule
        self.deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                                 scaling=2, padding='same', routings=3)

        # Layer 4: Convolutional Capsule: 1x1
        self.seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, stride=1, padding=0,
                                         routings=3)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = Length(num_classes=num_classes, seg=True)

        self.decoder = Decoder(16, input_channels)

    def forward(self, x):
        x = self.conv1(x)  # (N, C, H, W)

        conv1_reshaped = x.unsqueeze(dim=1)  # Reshape layer to be 1 capsule x [filters] atoms (N, 1, C, H, W)

        primary_caps = self.primary_caps(x)
        conv_cap_2_1 = self.conv_cap_2_1(primary_caps)
        conv_cap_2_2 = self.conv_cap_2_2(conv_cap_2_1)
        conv_cap_3_1 = self.conv_cap_3_1(conv_cap_2_2)
        conv_cap_3_2 = self.conv_cap_3_2(conv_cap_3_1)
        conv_cap_4_1 = self.conv_cap_4_1(conv_cap_3_2)
        conv_cap_4_2 = self.conv_cap_4_2(conv_cap_4_1)

        deconv_cap_1_1 = self.deconv_cap_1_1(conv_cap_4_2)
        up_1 = torch.cat([deconv_cap_1_1, conv_cap_3_1], axis=-2)
        deconv_cap_1_2 = self.deconv_cap_1_2(up_1)

        deconv_cap_2_1 = self.deconv_cap_2_1(deconv_cap_1_2)
        up_2 = torch.cat([deconv_cap_2_1, conv_cap_2_1])
        deconv_cap_2_2 = self.deconv_cap_3_1(up_2)

        deconv_cap_3_1 = self.deconv_cap_3_1(deconv_cap_2_2)
        up_3 = torch.cat([deconv_cap_3_1, conv1_reshaped])
        seg_caps = self.seg_caps(up_3)

        out_seg = self.out_seg(seg_caps)

        if self.training:
            decoded = self.decoder(seg_caps)
            return out_seg, decoded
        else:
            return out_seg
