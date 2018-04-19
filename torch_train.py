import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from lib.tiles import ImageSlicer
from lib.torch import albunet, linknet, unet11, unet16

import cv2
import os.path
import argparse
import pandas as pd

from lib.torch.gcn import GCN
from lib.torch.tiramisu import FCDenseNet67
from lib.torch.torch_losses import DiceLoss, BCEWithLogitsLossAndJaccard, JaccardLoss, JaccardScore
from lib.torch.unet import UNet
from sklearn.model_selection import train_test_split
from lib.torch import common as T

from tqdm import tqdm

from plot import plot_train_history

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in os.listdir(dirname)]


def to_float_tensor(img: np.ndarray):
    # .copy() because RuntimeError: some of the strides of a given numpy array are negative.
    #  This is currently not supported, but will be added in future releases.
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
    return tensor


def normalize_image(x: np.ndarray):
    x = x.astype(np.float32, copy=True)
    x /= 127.5
    x -= 1.
    return x


class SimpleDataset(Dataset):
    def __init__(self, images, masks):
        self.images = [to_float_tensor(i) for i in images]
        self.masks = [to_float_tensor(m) for m in masks]

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)


def get_dataset(dataset_name, dataset_dir, grayscale, patch_size):
    dataset_name = dataset_name.lower()

    if dataset_name == 'dsb2018':
        images = find_in_dir(os.path.join(dataset_dir, 'images'))
        images = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR) for fname in images]
        images = [normalize_image(i) for i in images]
        if grayscale:
            images = [np.expand_dims(m, axis=-1) for m in images]

        masks = find_in_dir(os.path.join(dataset_dir, 'masks'))
        masks = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in masks]
        masks = [np.expand_dims(m, axis=-1) for m in masks]
        masks = [np.float32(m > 0) for m in masks]

        patch_images = []
        patch_masks = []
        for image, mask in zip(images, masks):
            slicer = ImageSlicer(image.shape, patch_size, patch_size // 2)

            patch_images.extend(slicer.split(image))
            patch_masks.extend(slicer.split(mask))

        return patch_images, patch_masks

    raise ValueError(dataset_name)


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate)

    if optimizer_name == 'rms':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)

    if optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)

    raise ValueError(optimizer_name)


def get_loss(loss):
    loss = loss.lower()

    if loss == 'bce':
        return torch.nn.BCEWithLogitsLoss()

    if loss == 'dice':
        return DiceLoss()

    if loss == 'jaccard':
        return JaccardLoss()

    if loss == 'bce_jaccard':
        return BCEWithLogitsLossAndJaccard(jaccard_weight=1)

    raise ValueError(loss)


def get_model(model_name, num_classes, patch_size):
    model_name = str.lower(model_name)

    if model_name == 'unet':
        return UNet(num_classes=num_classes)

    if model_name == 'unet11':
        return unet11.UNet11(num_classes=num_classes,pretrained=True)

    if model_name == 'unet16':
        return unet16.UNet16(num_classes=num_classes,pretrained=True)

    if model_name == 'linknet34':
        return linknet.LinkNet34(num_classes=num_classes,pretrained=True)

    if model_name == 'albunet':
        return albunet.AlbuNet(num_classes=num_classes,pretrained=True)

    if model_name == 'tiramisu67':
        return FCDenseNet67(n_classes=num_classes)

    if model_name == 'gcn':
        return GCN(num_classes=num_classes, input_size=patch_size)

    raise ValueError(model_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate(model: torch.nn.Module, criterion, metric, valid_loader):
    model.eval()
    losses = []
    metrics = []

    for inputs, targets in valid_loader:
        inputs = T.variable(inputs, volatile=True)
        targets = T.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])

        s = metric(outputs, targets)
        metrics.append(s.data[0])

    return np.mean(losses), np.mean(metrics)


def run_train_session(model_name: str, optimizer: str, loss, learning_rate: float, epochs: int, dataset_name: str, dataset_dir: str, experiment_dir: str,
                      experiment: str, grayscale: bool, patch_size: int, batch_size: int):
    np.random.seed(42)

    x, y = get_dataset(dataset_name, dataset_dir, grayscale=grayscale, patch_size=patch_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

    trainloader = DataLoader(SimpleDataset(x_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    validloader = DataLoader(SimpleDataset(x_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=True)

    model = get_model(model_name, num_classes=1, patch_size=patch_size)
    model = model.cuda()
    print('Training', model_name, 'Number of parameters', count_parameters(model))

    optim = get_optimizer(optimizer, model.parameters(), learning_rate)
    criterion = get_loss(loss).cuda()
    jaccard_metric = JaccardScore().cuda()
    report_each = 10

    train_losses = []
    valid_losses = []
    train_metric = []
    valid_metric = []
    best_loss = np.inf

    cudnn.benchmark = True

    for epoch in range(epochs):  # loop over the dataset multiple times
        trn_loss = []
        trn_metric = []
        model.train()

        with tqdm(total=len(trainloader) * batch_size) as tq:
            tq.set_description('Epoch {}, lr {}'.format(epoch, learning_rate))

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                x, y = data

                # wrap them in Variable
                x, y = T.variable(x), T.variable(y)

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(x)
                loss = criterion(outputs, y)

                bs = x.size(0)
                (bs * loss).backward()

                optim.step()

                # Compute metrics
                jaccard_score = jaccard_metric(outputs, y)

                trn_loss.append(loss.data[0])
                trn_metric.append(jaccard_score.data[0])

                tq.update(bs)
                mean_loss = np.mean(trn_loss[-report_each:])
                mean_jaccard = np.mean(trn_metric[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss), jaccard='{:.3f}'.format(mean_jaccard))

        val_loss, val_jaccard = validate(model, criterion, jaccard_metric, validloader)
        trn_loss = np.mean(trn_loss)
        trn_metric = np.mean(trn_metric)

        valid_losses.append(val_loss)
        train_losses.append(trn_loss)
        train_metric.append(trn_metric)
        valid_metric.append(val_jaccard)
        print('loss=%.3f jaccard=%.3f val_loss=%.3f val_jaccard=%.3f' % (trn_loss, trn_metric, val_loss, val_jaccard))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch+1,
                'loss' : best_loss,
            }, os.path.join(experiment_dir, f'{model_name}_checkpoint.pth'))

    print('Training is finished...')

    pd.DataFrame.from_dict({'val_loss': valid_losses, 'loss': train_losses, 'val_jaccard': valid_metric, 'jaccard': train_metric})\
                .to_csv(os.path.join(experiment_dir, experiment + '.csv'), index=False)

    # plot_train_history(experiment, [train_losses, valid_metric],[train_metric, valid_metric])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--grayscale', action='store_true', help='Whether to use grayscale image instead of RGB')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of the model')
    parser.add_argument('-p', '--patch-size', type=int, default=224)
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-l', '--loss', type=str, default='bce', help='Target loss')
    parser.add_argument('-o', '--optimizer', default='SGD', help='Name of the optimizer')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to use for training.')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Root directory where datasets are located.')
    parser.add_argument('-s', '--steps', type=int, default=128, help='Steps per epoch')
    parser.add_argument('-x', '--experiment', type=str, help='Name of the experiment')

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = 'torch_%s_%d_%s_%s' % (args.model, args.patch_size, 'gray' if args.grayscale else 'rgb', args.loss)

    experiment_dir = os.path.join('experiments', args.experiment)
    os.makedirs(experiment_dir,exist_ok=True)
    with open(os.path.join(experiment_dir, 'arguments.txt'), 'w') as f:
        f.write(' '.join(sys.argv[1:]))

    run_train_session(model_name=args.model,
                      dataset_name=args.dataset,
                      dataset_dir=args.data_dir,
                      patch_size=args.patch_size,
                      batch_size=args.batch_size,
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate,
                      experiment_dir=experiment_dir,
                      experiment=args.experiment,
                      grayscale=args.grayscale,
                      loss=args.loss,
                      epochs=args.epochs)



if __name__ == '__main__':
    main()
