import numpy as np

np.random.seed(42)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import os.path
import argparse
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras_losses import dice_loss, jaccard_loss, bce_jaccard_loss
from models.unet import ZF_UNET
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in os.listdir(dirname)]


def get_dataset(dataset_name, dataset_dir, grayscale):
    dataset_name = dataset_name.lower()

    if dataset_name == 'dsb2018':
        images = find_in_dir(os.path.join(dataset_dir, 'images'))
        masks = find_in_dir(os.path.join(dataset_dir, 'masks'))

        if grayscale:
            x = np.array([np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=-1) for fname in images])
        else:
            x = np.array([cv2.imread(fname, cv2.IMREAD_COLOR) for fname in images])

        y = np.array([np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=-1) for fname in masks])
        return x, y

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
        return KTF.binary_crossentropy

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

    raise ValueError(model_name)


def create_session(gpu_fraction):
    print('Setting GPU memory usage %d%%' % int(gpu_fraction * 100))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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

    experiment_dir = args.experiment
    os.makedirs(experiment_dir, exist_ok=True)

    x, y = get_dataset(args.dataset, args.data_dir, args.grayscale)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

    optim = get_optimizer(args.optimizer, args.learning_rate)
    loss = get_loss(args.loss)
    model = get_model(args.model, args.patch_size, args.grayscale)
    model.compile(optimizer=optim, loss=loss)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            experiment_dir,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True),
    ]

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  shuffle=True,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  verbose=2
                  )

    print('Training is finished...')

    pd.DataFrame(h.history).to_csv(os.path.join(experiment_dir, args.experiment + '.csv'), index=False)

    # utils.plot_train_history(h.history, model.name,
    #                          [['loss', 'val_loss']],
    #                          optimizer=optim,
    #                          figure_filename=os.path.join(experiment_dir, 'train_history.png'))


if __name__ == '__main__':
    main()
