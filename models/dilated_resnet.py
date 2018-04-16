from keras.layers import Input, Convolution2D
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras.models import Model


def DilatedResnet(dropout_val=0.1,
                  filters=32,
                  batch_norm=True,
                  patch_size=224,
                  activation='relu',
                  input_channels=3,
                  output_classes=1):

    input = Input((patch_size, patch_size, input_channels))

    conv1 = Convolution2D(filters, (3, 3), padding='same', activation=activation)(input)

    conv2_input = concatenate([input, conv1])
    conv2 = Convolution2D(filters, (3, 3), dilation_rate=2, padding='same', activation=activation)(conv2_input)

    conv3_input = concatenate([input, conv1, conv2])
    conv3 = Convolution2D(filters, (3, 3), dilation_rate=3, padding='same', activation=activation)(conv3_input)

    conv4_input = concatenate([input, conv1, conv2, conv3])
    conv4 = Convolution2D(filters, (3, 3), dilation_rate=4, padding='same', activation=activation)(conv4_input)

    conv5_input = concatenate([input, conv1, conv2, conv3, conv4])
    conv5 = Convolution2D(filters, (3, 3), dilation_rate=5, padding='same', activation=activation)(conv5_input)

    final_input = concatenate([input, conv1, conv2, conv3, conv4, conv5])
    final = Convolution2D(output_classes, (3, 3), padding='same')(final_input)

    if output_classes == 1:
        final = Activation('sigmoid')(final)
    else:
        final = Activation('softmax')(final)

    model = Model(input, final, name='DilatedResnet')
    return model
