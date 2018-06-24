'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module, Parameter

# Pytorch convention is    N x Atoms x Capsules x Height x Width, (NACHW)
# Tensorflow convention is N x Height x Width x Capsules x Atoms, (NHWCA)

ATOMS_DIM = 1
CAPS_DIM = 2


class Length(nn.Module):
    def __init__(self, num_classes, seg=True):
        super(Length, self).__init__()
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def forward(self, inputs: Tensor):
        if len(inputs.size()) == 5:
            assert inputs.size(CAPS_DIM) == 1, 'Error: Must have num_capsules = 1 going into Length'
            inputs = torch.squeeze(inputs, dim=CAPS_DIM)
        return torch.expand_dims(torch.norm(inputs, dim=ATOMS_DIM), dim=ATOMS_DIM)


class Mask(Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, inputs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs

            mask = K.expand_dims(mask, -1)
            if input.get_shape().ndims == 3:
                masked = K.batch_flatten(mask * input)
            else:
                masked = mask * input

        else:
            if inputs.get_shape().ndims == 3:
                x = K.sqrt(K.sum(K.square(inputs), -1))
                mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
                masked = K.batch_flatten(K.expand_dims(mask, -1) * inputs)
            else:
                masked = inputs

        return masked


class ConvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, in_capsules, in_atoms, num_capsule, num_atoms, strides=1, padding=0, routings=3):
        super(ConvCapsuleLayer, self).__init__()
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings

        self.input_num_capsule = in_capsules
        self.input_num_atoms = in_atoms
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))


        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                        self.input_num_atoms, self.num_capsule * self.num_atoms], name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms], name='b')

    def forward(self, input_tensor: Tensor):
        assert len(input_tensor.size()) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"

        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        # conv = F.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
        #                 padding=self.padding)

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = conv.size()
        _, conv_height, conv_width, _ = votes_shape # Needs to fix

        votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [conv_height.value, conv_width.value, 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations


class DeconvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, in_capsules, in_atoms, num_capsule, num_atoms, scaling=2, upsamp_type='deconv', padding='same', routings=3):
        super(DeconvCapsuleLayer, self).__init__()
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.scaling = scaling
        self.upsamp_type = upsamp_type
        self.padding = padding
        self.routings = routings

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        if self.upsamp_type == 'subpix':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms,
                                            self.num_capsule * self.num_atoms * self.scaling * self.scaling],
                                     initializer=self.kernel_initializer,
                                     name='W')
        elif self.upsamp_type == 'resize':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms, self.num_capsule * self.num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        elif self.upsamp_type == 'deconv':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.num_capsule * self.num_atoms, self.input_num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        else:
            raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms], name='b')

        self.built = True

    def forward(self, input_tensor):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[1] * input_shape[0], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        if self.upsamp_type == 'resize':
            upsamp = K.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
            outputs = K.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding, data_format='channels_last')
        elif self.upsamp_type == 'subpix':
            conv = K.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
                            data_format='channels_last')
            outputs = tf.depth_to_space(conv, self.scaling)
        else:
            batch_size = input_shape[1] * input_shape[0]

            # Infer the dynamic output shape:
            out_height = deconv_length(self.input_height, self.scaling, self.kernel_size, self.padding)
            out_width = deconv_length(self.input_width, self.scaling, self.kernel_size, self.padding)
            output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)

            outputs = K.conv2d_transpose(input_tensor_reshaped, self.W, output_shape, (self.scaling, self.scaling),
                                         padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(outputs)
        _, conv_height, conv_width, _ = outputs.get_shape()

        votes = K.reshape(outputs, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                    self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations


def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits, dim=-1)
        preactivate_unrolled = route * votes_trans
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
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return K.cast(activations.read(num_routing - 1), dtype='float32')


def _squash(input_tensor: Tensor):
    norm = torch.norm(input_tensor, 2, dim=ATOMS_DIM, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


class DecoderModule(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.recon_1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=1)
        self.recon_2 = nn.Conv2d(64, 128, kernel_size=1)
        self.out_recon = nn.Conv2d(128, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.recon_1(x)
        x = F.relu(x, True)
        x = self.recon_2(x)
        x = F.relu(x, True)
        x = self.out_recon(x)
        x = F.sigmoid(x)
        return x


class CapsNetR3(Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.n_classes = 2

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=2)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        self.primary_caps = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=1, in_atoms=16, num_capsule=2, num_atoms=16, strides=2, routings=1)

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=2, in_atoms=16, num_capsule=4, num_atoms=16, strides=1, routings=3)

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=4, in_atoms=16, num_capsule=4, num_atoms=32, strides=2, routings=3)

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=4, in_atoms=32, num_capsule=8, num_atoms=32, strides=1, routings=3)

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=8, in_atoms=32, num_capsule=8, num_atoms=64, strides=2, routings=3)

        # Layer 4: Convolutional Capsule
        self.conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=8, in_atoms=64, num_capsule=8, num_atoms=32, strides=1, routings=3)

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, in_capsules=8, in_atoms=32, num_capsule=8, num_atoms=32, upsamp_type='deconv', scaling=2, routings=3)

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, padding=2, in_capsules=16, in_atoms=32, num_capsule=4, num_atoms=16, strides=1, routings=3)

        # Layer 3 Up: Deconvolutional Capsule
        self.deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, in_capsules= num_capsule=2, num_atoms=16, upsamp_type='deconv', scaling=2, routings=3)

        # Layer 4: Convolutional Capsule: 1x1
        self.seg_caps = ConvCapsuleLayer(kernel_size=1, padding=0, num_capsule=1, num_atoms=16, strides=1, routings=3)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = Length(num_classes=n_classes, seg=True)

        self.reconstruction = DecoderModule(32)

    def forward(self, x, y):
        """
        Capsules tensor order: # N x Atoms x Capsules x Height x Width, (NACHW)
        :param x: Tensor of NxCxHxW
        :return:
        """
        conv1 = self.conv1(x)
        conv1 = F.relu(conv1, True)
        # Reshape layer to be 1 capsule x [filters] atoms
        conv1 = torch.unsqueeze(conv1, dim=CAPS_DIM)  # NA1HW

        # Tensorflow convention is NHW1A

        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = self.primary_caps(conv1)
        conv_cap_2_1 = self.conv_cap_2_1(primary_caps)
        conv_cap_2_2 = self.conv_cap_2_2(conv_cap_2_1)
        conv_cap_3_1 = self.conv_cap_3_1(conv_cap_2_2)
        conv_cap_3_2 = self.conv_cap_3_2(conv_cap_3_1)
        conv_cap_4_1 = self.conv_cap_4_1(conv_cap_3_2)

        deconv_cap_1_1 = self.deconv_cap_1_1(conv_cap_4_1)

        # Skip connection. Concatenate on capsule dim
        up_1 = torch.cat(deconv_cap_1_1, conv_cap_3_1, axis=CAPS_DIM)

        deconv_cap_1_2 = self.deconv_cap_1_2(up_1)
        deconv_cap_2_1 = self.deconv_cap_2_1(deconv_cap_1_2)

        # Skip connection
        up_2 = torch.cat(deconv_cap_2_1, conv_cap_2_1, axis=CAPS_DIM)

        deconv_cap_2_2 = self.deconv_cap_2_2(up_2)
        deconv_cap_3_1 = self.deconv_cap_3_1(deconv_cap_2_2)

        # Skip connection
        up_3 = torch.cat(deconv_cap_3_1, conv1, dim=CAPS_DIM)

        # Layer 4: Convolutional Capsule: 1x1
        seg_caps = self.seg_caps(up_3)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = self.out_seg(seg_caps)

        if self.training:
            masked_by_y = seg_caps * y
            reconstructed = self.reconstruction(masked_by_y)
        else:
            reconstructed = self.mask(seg_caps)

        return out_seg, reconstructed
