from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AlphaDropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate


def double_conv_layer(x, size, dropout, batch_norm):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation('selu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation('selu')(conv)
    if dropout > 0:
        conv = AlphaDropout(dropout)(conv)
    return conv


def Selunet(dropout_val=0.2, batch_norm=True, patch_size=224, input_channels=3, output_classes=1,filters = 32):
    inputs = Input((patch_size, patch_size, input_channels))

    conv_224 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2 * filters, dropout_val, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4 * filters, dropout_val, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8 * filters, dropout_val, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16 * filters, dropout_val, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32 * filters, dropout_val, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14])
    up_conv_14 = double_conv_layer(up_14, 16 * filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28])
    up_conv_28 = double_conv_layer(up_28, 8 * filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56])
    up_conv_56 = double_conv_layer(up_56, 4 * filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112])
    up_conv_112 = double_conv_layer(up_112, 2 * filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224])
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(output_classes, (1, 1))(up_conv_224)

    if output_classes == 1:
        conv_final = Activation('sigmoid')(conv_final)
    else:
        conv_final = Activation('softmax')(conv_final)

    model = Model(inputs, conv_final, name="Selunet")
    return model
