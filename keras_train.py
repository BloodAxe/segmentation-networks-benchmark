import numpy as np

from lib.keras.keras_losses import dice_loss, jaccard_loss
from lib.keras.dilated_resnet import DilatedResnet
from lib.keras.dilated_unet import DilatedUnet
from lib.keras.linknet import LinkNet
from lib.keras.selunet import Selunet
from lib.keras.tiramisu67 import Tiramisu67
from lib.keras.zf_unet import ZF_UNET
from lib.tiles import ImageSlicer


import cv2
import os.path
import argparse
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
from keras.applications import imagenet_utils


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in os.listdir(dirname)]


def normalize_image(x: np.ndarray):
    x = x.astype(np.float32, copy=True)
    x /= 127.5
    x -= 1.
    return x


def get_dataset(dataset_name, dataset_dir, grayscale, patch_size):
    dataset_name = dataset_name.lower()

    if dataset_name == 'dsb2018':
        images = find_in_dir(os.path.join(dataset_dir, dataset_name, 'images'))
        images = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR) for fname in images]
        images = [normalize_image(i) for i in images]
        if grayscale:
            images = [np.expand_dims(m, axis=-1) for m in images]

        masks = find_in_dir(os.path.join(dataset_dir, dataset_name, 'masks'))
        masks = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in masks]
        masks = [np.expand_dims(m, axis=-1) for m in masks]
        masks = [np.float32(m > 0) for m in masks]

        patch_images = []
        patch_masks = []
        for image, mask in zip(images, masks):
            slicer = ImageSlicer(image.shape, patch_size, patch_size//2)

            patch_images.extend(slicer.split(image))
            patch_masks.extend(slicer.split(mask))

        return np.array(patch_images), np.array(patch_masks)

    raise ValueError(dataset_name)


def get_optimizer(optimizer_name, learning_rate):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return SGD(lr=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)

    if optimizer_name == 'rms':
        return RMSprop(lr=learning_rate)

    if optimizer_name == 'adam':
        return Adam(lr=learning_rate, decay=1e-4)

    raise ValueError(optimizer_name)


def get_loss(loss):
    loss = loss.lower()

    if loss == 'bce':
        return 'binary_crossentropy'

    if loss == 'dice':
        return dice_loss

    if loss == 'jaccard':
        return jaccard_loss

    if loss == 'bce_jaccard':
        return bce_jaccard_loss

    raise ValueError(loss)


def get_model(model_name, patch_size, grayscale):
    input_channels = 1 if grayscale else 3
    model_name = str.lower(model_name)

    if model_name == 'zf_unet':
        return ZF_UNET(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    if model_name == 'selunet':
        return Selunet(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    if model_name == 'dilated_unet':
        return DilatedUnet(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    if model_name == 'dilated_resnet':
        return DilatedResnet(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    if model_name == 'linknet':
        return LinkNet(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    if model_name == 'tiramisu67':
        return Tiramisu67(patch_size=patch_size, input_channels=input_channels, output_classes=1)

    raise ValueError(model_name)


def create_session(gpu_fraction):
    print('Setting GPU memory usage %d%%' % int(gpu_fraction * 100))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def run_train_session(model, optimizer, loss, learning_rate, epochs, dataset_name, dataset_dir, experiment, grayscale, patch_size, batch_size):
    np.random.seed(42)

    os.makedirs(experiment, exist_ok=True)

    x, y = get_dataset(dataset_name, dataset_dir, grayscale=grayscale, patch_size=patch_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

    optim = get_optimizer(optimizer, learning_rate)
    loss = get_loss(loss)
    model = get_model(model, patch_size, grayscale)
    model.compile(optimizer=optim, loss=loss, metrics=[jaccard_coef])
    model.summary()

    callbacks = [
        ModelCheckpoint(
            os.path.join(experiment, experiment + '.h5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True),
    ]

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  shuffle=True,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  verbose=2
                  )

    print('Training is finished...')

    pd.DataFrame(h.history).to_csv(os.path.join(experiment, experiment + '.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--grayscale', action='store_true', help='Whether to use grayscale image instead of RGB')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of the model')
    parser.add_argument('-p', '--patch-size', type=int, default=224)
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('-l', '--loss', type=str, default='bce', help='Target loss')
    parser.add_argument('-o', '--optimizer', default='SGD', help='Name of the optimizer')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Epoch to run')
    parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to use for training.')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Root directory where datasets are located.')
    parser.add_argument('-s', '--steps', type=int, default=128, help='Steps per epoch')
    parser.add_argument('-x', '--experiment', type=str, help='Name of the experiment')
    parser.add_argument('--gpu-fraction', type=float, default=None, help='Sets maximum GPU memory fraction that process can use')

    args = parser.parse_args()

    if args.gpu_fraction is not None:
        KTF.set_session(create_session(args.gpu_fraction))

    if args.experiment is None:
        args.experiment = '%s_%d_%s_%s' % (args.model, args.patch_size, 'gray' if args.grayscale else 'rgb', args.loss)

    run_train_session(model=args.model,
                      dataset_name=args.dataset,
                      dataset_dir=args.data_dir,
                      patch_size=args.patch_size,
                      batch_size=args.batch_size,
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate,
                      experiment=args.experiment,
                      grayscale=args.grayscale,
                      loss=args.loss,
                      epochs=args.epochs)


if __name__ == '__main__':
    main()
