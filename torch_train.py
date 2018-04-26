import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchvision.datasets import CocoDetection
from torchvision import transforms as trans
from torchvision.utils import make_grid

from lib.tiles import ImageSlicer
from lib.torch import albunet, linknet, unet11, unet16

import cv2
import os.path
import argparse
import pandas as pd

from lib.torch.ImageMaskDataset import ImageMaskDataset
from lib.torch.common import to_float_tensor, show_landmarks_batch
from lib.torch.factorized_unet11 import FactorizedUNet11
from lib.torch.gcn import GCN
from lib.torch.psp_net import PSPNet
from lib.torch.seg_net import SegCaps
from lib.torch.tiramisu import FCDenseNet67
from lib.torch.torch_losses import DiceLoss, BCEWithLogitsLossAndJaccard, JaccardLoss, JaccardScore
from lib.torch.unet import UNet, FactorizedUNet
from sklearn.model_selection import train_test_split
from lib.torch import common as T
from lib import augmentations as aug
from tqdm import tqdm

from plot import plot_train_history

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in os.listdir(dirname)]


def read_rgb(fname):
    x = cv2.imread(fname, cv2.IMREAD_COLOR)
    return x


def read_gray(fname):
    x = np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=-1)
    return x


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

    if dataset_name == 'coco':
        train_transform = trans.Compose([
            trans.RandomCrop(patch_size),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            trans.RandomGrayscale(),
            trans.ToTensor(),
        ])

        test_transform = trans.Compose([
            trans.CenterCrop(patch_size),
            trans.transforms.ToTensor(),
        ])

        num_classes = 182
        return CocoDetection(root=os.path.join(dataset_dir, 'train2014'),
                             annFile=os.path.join(dataset_dir, 'annotations_trainval2014', 'instances_train2014.json'),
                             transform=train_transform), \
               CocoDetection(root=os.path.join(dataset_dir, 'val2014'),
                             annFile=os.path.join(dataset_dir, 'annotations_trainval2014', 'instances_val2014.json'),
                             transform=test_transform), \
               num_classes

    if dataset_name == 'inria':
        x = find_in_dir(os.path.join(dataset_dir, 'images'))
        y = find_in_dir(os.path.join(dataset_dir, 'gt'))

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.1)

        train_transform = aug.Sequential([
            # aug.RandomCrop(224),
            aug.VerticalFlip(),
            aug.HorizontalFlip(),
            # aug.ImageOnly(trans.RandomGrayscale()),
            aug.ImageOnly(aug.RandomBrightness()),
            aug.ImageOnly(aug.RandomContrast()),
            aug.ImageOnly(aug.ScaleImage()),
            aug.MaskOnly(aug.MakeBinary()),
            aug.ToTensors()
        ])

        test_transform = aug.Sequential([
            # aug.CenterCrop(patch_size, patch_size),
            aug.ImageOnly(aug.ScaleImage()),
            aug.MaskOnly(aug.MakeBinary()),
            aug.ToTensors()
        ])

        train = ImageMaskDataset(x_train, y_train, image_loader=read_rgb, target_loader=read_gray, transform=train_transform, load_in_ram=False)
        test = ImageMaskDataset(x_test, y_test, image_loader=read_rgb, target_loader=read_gray, transform=test_transform, load_in_ram=False)
        num_classes = 1
        return train, test, num_classes

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

        x_train, x_test, y_train, y_test = train_test_split(patch_images, patch_masks, random_state=1234, test_size=0.2)

        num_classes = 1
        return SimpleDataset(x_train, y_train), SimpleDataset(x_test, y_test), num_classes

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
        return torch.nn.BCEWithLogitsLoss(), [JaccardScore()]

    if loss == 'dice':
        return DiceLoss(), [JaccardScore()]

    if loss == 'jaccard':
        return JaccardLoss(), [JaccardScore()]

    if loss == 'bce_jaccard':
        return BCEWithLogitsLossAndJaccard(jaccard_weight=1), [JaccardScore()]

    if loss == 'nlll2d':
        return torch.nn.NLLLoss2d(), []

    raise ValueError(loss)


def get_model(model_name, num_classes, patch_size):
    model_name = str.lower(model_name)

    if model_name == 'unet':
        return UNet(num_classes=num_classes)

    if model_name == 'factorized_unet':
        return FactorizedUNet(num_classes=num_classes)

    if model_name == 'unet11':
        return unet11.UNet11(num_classes=num_classes, pretrained=True)

    if model_name == 'factorized_unet11':
        return FactorizedUNet11(num_classes=num_classes, pretrained=True)

    if model_name == 'unet16':
        return unet16.UNet16(num_classes=num_classes, pretrained=True)

    if model_name == 'linknet34':
        return linknet.LinkNet34(num_classes=num_classes, pretrained=True)

    if model_name == 'albunet':
        return albunet.AlbuNet(num_classes=num_classes, pretrained=True)

    if model_name == 'tiramisu67':
        return FCDenseNet67(n_classes=num_classes)

    if model_name == 'gcn':
        return GCN(num_classes=num_classes, input_size=patch_size)

    if model_name == 'psp_net':
        return PSPNet(num_classes=num_classes, pretrained=True, use_aux=False)

    if model_name == 'seg_caps':
        return SegCaps(num_classes=num_classes)

    raise ValueError(model_name)




def validate(model: torch.nn.Module, criterion, metrics, valid_loader):
    model.eval()
    losses = []

    metrics_scores = [[]] * len(metrics)

    for inputs, targets in valid_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])

        for metric_index, metric in enumerate(metrics):
            score = metric(outputs, targets)
            metrics_scores[metric_index].append(score.data[0])

    return losses, metrics_scores


def run_train_session_binary(model_name: str, optimizer: str, loss, learning_rate: float, epochs: int, dataset_name: str, dataset_dir: str, experiment_dir: str,
                             experiment: str, grayscale: bool, patch_size: int, batch_size: int, workers: int, resume: bool):
    np.random.seed(42)

    trainset, validset, num_classes = get_dataset(dataset_name, dataset_dir, grayscale=grayscale, patch_size=patch_size)
    print('Train set size', len(trainset))
    print('Valid set size', len(validset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, pin_memory=True)

    show_landmarks_batch(next(trainloader.__iter__()))
    show_landmarks_batch(next(validloader.__iter__()))

    model = get_model(model_name, num_classes=num_classes, patch_size=patch_size)
    model = model.cuda()
    print('Training', model_name, 'Number of parameters', count_parameters(model))

    optim = get_optimizer(optimizer, model.parameters(), learning_rate)
    criterion, metrics = get_loss(loss)

    criterion = criterion.cuda()
    metrics = [m.cuda() for m in metrics]

    start_epoch = 0
    report_each = 10
    best_loss = np.inf

    cudnn.benchmark = True

    train_history = {
        'epoch': [],
        'loss': [],
        'val_loss': []
    }

    checkpoint_filename = os.path.join(experiment_dir, f'{model_name}_checkpoint.pth')
    if resume:
        checkpoint = torch.load(checkpoint_filename)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model'])
        print('Resuming training from epoch', start_epoch, ' and loss', best_loss)

    for m in metrics:
        train_history[str(m)] = []
        train_history['val_' + str(m)] = []

    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        trn_losses = []
        trn_metric_scores = [[]] * len(metrics)

        model.train()

        with tqdm(total=len(trainloader) * batch_size) as tq:
            tq.set_description('Epoch {}, lr {}'.format(epoch, learning_rate))

            for i, (x, y) in enumerate(trainloader, 0):
                x, y = x.cuda(), y.cuda()

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(x)
                loss = criterion(outputs, y)

                bs = x.size(0)
                (bs * loss).backward()

                optim.step()

                # Compute metrics
                for metric_index, metric in enumerate(metrics):
                    score = metric(outputs, y)
                    trn_metric_scores[metric_index].append(score.data.item())

                trn_losses.append(loss.data.item())

                tq.update(bs)
                mean_loss = np.mean(trn_losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))

        val_losses, val_metric_scores = validate(model, criterion, metrics, validloader)
        val_loss = np.mean(val_losses)
        trn_loss = np.mean(trn_losses)

        train_history['epoch'].append(epoch)
        train_history['loss'].append(trn_loss)
        train_history['val_loss'].append(val_loss)
        for m, scores in zip(metrics, trn_metric_scores):
            train_history[str(m)].append(np.mean(scores))
        for m, scores in zip(metrics, val_metric_scores):
            train_history['val_' + str(m)].append(np.mean(scores))

        trn_metric_str = ['%s=%.3f' % i for i in [(str(m), np.mean(scores)) for m, scores in zip(metrics, trn_metric_scores)]]
        val_metric_str = ['val_%s=%.3f' % i for i in [(str(m), np.mean(scores)) for m, scores in zip(metrics, val_metric_scores)]]

        print('loss=%.3f val_loss=%.3f' % (trn_loss, val_loss), ' '.join(trn_metric_str + val_metric_str))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, checkpoint_filename)

    print('Training is finished...')

    pd.DataFrame.from_dict(train_history).to_csv(os.path.join(experiment_dir, experiment + '.csv'),
                                                 index=False,
                                                 mode='w' if resume is None else 'a',
                                                 header=resume is False)


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
    parser.add_argument('-w', '--workers', default=0, type=int, help='Num workers')
    parser.add_argument('-r', '--resume', action='store_true')

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = 'torch_%s_%s_%d_%s_%s' % (args.dataset, args.model, args.patch_size, 'gray' if args.grayscale else 'rgb', args.loss)

    experiment_dir = os.path.join('experiments', args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'arguments.txt'), 'w') as f:
        f.write(' '.join(sys.argv[1:]))

    run_train_session_binary(model_name=args.model,
                             dataset_name=args.dataset,
                             dataset_dir=args.data_dir,
                             patch_size=args.patch_size,
                             batch_size=args.batch_size,
                             optimizer=args.optimizer,
                             learning_rate=args.learning_rate,
                             experiment_dir=os.path.normpath(experiment_dir),
                             resume=args.resume,
                             experiment=args.experiment,
                             grayscale=args.grayscale,
                             loss=args.loss,
                             epochs=args.epochs,
                             workers=args.workers)


if __name__ == '__main__':
    main()
