import argparse
import sys

import numpy as np
import os.path
import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.torch.common import show_landmarks_batch, count_parameters
from lib.torch.datasets.Inria import INRIA
from lib.torch.datasets.coco import COCO
from lib.torch.datasets.dsb2018 import DSB2018
from lib.torch.models import linknet, albunet, unet16, unet11
from lib.torch.models.factorized_unet11 import FactorizedUNet11
from lib.torch.models.gcn import GCN
from lib.torch.models.psp_net import PSPNet
from lib.torch.models.seg_net import SegCaps, SegCapsLoss
from lib.torch.models.tiramisu import FCDenseNet67
from lib.torch.models.unet import UNet, FactorizedUNet
from lib.torch.models.zf_unet import ZF_UNET
from lib.torch.torch_losses import DiceLoss, BCEWithLogitsLossAndJaccard, JaccardLoss, JaccardScore

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def get_dataset(dataset_name, dataset_dir, grayscale, patch_size):
    dataset_name = dataset_name.lower()

    if dataset_name == 'coco':
        return COCO(dataset_dir, grayscale, patch_size)

    if dataset_name == 'inria':
        return INRIA(dataset_dir, grayscale, patch_size)

    if dataset_name == 'dsb2018':
        return DSB2018(dataset_dir, grayscale, patch_size)

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

    if loss == 'caps_loss':
        return SegCapsLoss(), [JaccardScore()]

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

    if model_name == 'zf_unet':
        return ZF_UNET(num_classes=num_classes)

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


def run_train_session_binary(model_name: str, optimizer: str, loss, learning_rate: float, epochs: int, dataset_name: str, dataset_dir: str, experiment_dir: str,
                             experiment: str, grayscale: bool, patch_size: int, batch_size: int, workers: int, resume: bool):
    np.random.seed(42)

    trainset, validset, num_classes = get_dataset(dataset_name, dataset_dir, grayscale=grayscale, patch_size=patch_size)
    print('Train set size', len(trainset))
    print('Valid set size', len(validset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

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
                if model_name == 'seg_caps':
                    loss = criterion(outputs, (y, x))  # For segcaps we need both x and y to compute reconstruction loss term
                else:
                    loss = criterion(outputs, y)

                bs = x.size(0)
                (bs * loss).backward()

                optim.step()

                # Compute metrics
                for metric_index, metric in enumerate(metrics):
                    if model_name == 'seg_caps':
                        score = metric(outputs[0], y)  # We interested in segmentation output for computing metric
                    else:
                        score = metric(outputs, y)

                    trn_metric_scores[metric_index].append(score.data.item())

                trn_losses.append(loss.data.item())

                tq.update(bs)
                mean_loss = np.mean(trn_losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))

        # Run validation
        val_metric_scores = [[]] * len(metrics)
        val_losses = []
        for x, y in validloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            if model_name == 'seg_caps':
                loss = criterion(outputs, (y, x))
            else:
                loss = criterion(outputs, y)

            val_losses.append(loss.data.item())

            for metric_index, metric in enumerate(metrics):
                if model_name == 'seg_caps':
                    score = metric(outputs[0], y)  # We interested in segmentation output for computing metric
                else:
                    score = metric(outputs, y)
                val_metric_scores[metric_index].append(score.data.item())

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

    experiment_dir = os.path.join('experiments', args.dataset, args.loss, args.experiment)
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
