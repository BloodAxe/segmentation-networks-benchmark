import argparse
import os.path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from lib.common import count_parameters
from lib.datasets.Inria import INRIA
from lib.datasets.dsb2018 import DSB2018Sliced
from lib.losses import JaccardLoss, FocalLossBinary, BCEWithLogitsLossAndSmoothJaccard, BCEWithSigmoidLoss
from lib.metrics import JaccardScore, PixelAccuracy
from lib.models import linknet, unet16, unet11
from lib.models.afterburner import Afterburner
from lib.models.duc_hdc import ResNetDUCHDC, ResNetDUC
from lib.models.gcn152 import GCN152, GCN34
from lib.models.psp_net import PSPNet
from lib.models.tiramisu import FCDenseNet67
from lib.models.unet import UNet
from lib.models.zf_unet import ZF_UNET
from lib.train_utils import AverageMeter, auto_file
import torch_train as TT

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def train(model, loss, optimizer, dataloader, epoch: int, metrics={}, summary_writer=None):
    losses = AverageMeter()

    train_scores = {}
    for key, _ in metrics.items():
        train_scores[key] = AverageMeter()

    with torch.set_grad_enabled(True):
        model.train()
        n_batches = len(dataloader)
        with tqdm(total=n_batches) as tq:
            tq.set_description('Train')
            x = None
            y = None
            outputs = None
            batch_loss = None

            for batch_index, (x, y) in enumerate(dataloader):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(x)

                batch_loss = loss(outputs, y)

                batch_size = x.size(0)
                (batch_size * batch_loss).backward()

                optimizer.step()

                # Batch train end
                # Log train progress

                batch_loss_val = batch_loss.cpu().item()
                if summary_writer is not None:
                    summary_writer.add_scalar('train/batch/loss', batch_loss_val, epoch * n_batches + batch_index)

                    # Plot gradient absmax to see if there are any gradient explosions
                    grad_max = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_max = max(grad_max, param.grad.abs().max().cpu().item())
                    summary_writer.add_scalar('train/grad/global_max', grad_max, epoch * n_batches + batch_index)

                losses.update(batch_loss_val)

                for key, metric in metrics.items():
                    score = metric(outputs, y).cpu().item()
                    train_scores[key].update(score)

                    if summary_writer is not None:
                        summary_writer.add_scalar('train/batch/' + key, score, epoch * n_batches + batch_index)

                tq.set_postfix(loss='{:.3f}'.format(losses.avg), **train_scores)
                tq.update()

            # End of train epoch
            if summary_writer is not None:
                summary_writer.add_image('train/image', make_grid(x.cpu(), normalize=True), epoch)
                summary_writer.add_image('train/y_true', make_grid(y.cpu(), normalize=True), epoch)
                summary_writer.add_image('train/y_pred', make_grid(outputs.sigmoid().cpu(), normalize=True), epoch)
                summary_writer.add_scalar('train/epoch/loss', losses.avg, epoch)
                for key, value in train_scores.items():
                    summary_writer.add_scalar('train/epoch/' + key, value.avg, epoch)

                # Plot histogram of parameters after each epoch
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Plot weighs
                        param_data = param.data.cpu().numpy()
                        summary_writer.add_histogram('model/' + name, param_data, epoch, bins='doane')

                # for m in model.modules():
                #     if isinstance(m, nn.Conv2d):
                #         weights = m.weights.data.numpy()

            del x, y, outputs, batch_loss

    return losses, train_scores


def validate(model, loss, dataloader, epoch: int, metrics=dict(), summary_writer: SummaryWriter = None):
    losses = AverageMeter()

    valid_scores = {}
    for key, _ in metrics.items():
        valid_scores[key] = AverageMeter()

    with torch.set_grad_enabled(False):
        model.eval()

        n_batches = len(dataloader)
        with tqdm(total=len(dataloader)) as tq:
            tq.set_description('Validation')

            x = None
            y = None
            outputs = None
            batch_loss = None

            for batch_index, (x, y) in enumerate(dataloader):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                # forward + backward + optimize
                outputs = model(x)
                batch_loss = loss(outputs, y)

                # Log train progress

                batch_loss_val = batch_loss.cpu().item()
                if summary_writer is not None:
                    summary_writer.add_scalar('val/batch/loss', batch_loss_val, epoch * n_batches + batch_index)

                losses.update(batch_loss_val)

                for key, metric in metrics.items():
                    score = metric(outputs, y).cpu().item()
                    valid_scores[key].update(score)

                    if summary_writer is not None:
                        summary_writer.add_scalar('val/batch/' + key, score, epoch * n_batches + batch_index)

                tq.set_postfix(loss='{:.3f}'.format(losses.avg), **valid_scores)
                tq.update()

            if summary_writer is not None:
                summary_writer.add_image('val/image', make_grid(x.cpu(), normalize=True), epoch)
                summary_writer.add_image('val/y_true', make_grid(y.cpu(), normalize=True), epoch)
                summary_writer.add_image('val/y_pred', make_grid(outputs.sigmoid().cpu(), normalize=True), epoch)
                summary_writer.add_scalar('val/epoch/loss', losses.avg, epoch)
                for key, value in valid_scores.items():
                    summary_writer.add_scalar('val/epoch/' + key, value.avg, epoch)

            del x, y, outputs, batch_loss

    return losses, valid_scores


def save_snapshot(model: nn.Module, optimizer: Optimizer, loss: float, epoch: int, train_history: pd.DataFrame, snapshot_file: str):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_history': train_history.to_dict(),
        'args': ' '.join(sys.argv[1:])
    }, snapshot_file)


def restore_snapshot(model: nn.Module, optimizer: Optimizer, snapshot_file: str):
    checkpoint = torch.load(snapshot_file)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_history = pd.DataFrame.from_dict(checkpoint['train_history'])

    return start_epoch, train_history, best_loss


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
    parser.add_argument('-mem', '--memory', action='store_true')

    args = parser.parse_args()
    cudnn.benchmark = True

    if args.experiment is None:
        args.experiment = 'torch_%s_%s_afterburn_%d_%s_%s' % (args.dataset, args.model, args.patch_size, 'gray' if args.grayscale else 'rgb', args.loss)

    experiment_dir = os.path.join('experiments', args.dataset, args.loss, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    writer = SummaryWriter(comment=args.experiment)

    with open(os.path.join(experiment_dir, 'arguments.txt'), 'w') as f:
        f.write(' '.join(sys.argv[1:]))

    trainset, validset, num_classes = TT.get_dataset(args.dataset, args.data_dir, grayscale=args.grayscale, patch_size=args.patch_size, keep_in_mem=args.memory)
    print('Train set size', len(trainset))
    print('Valid set size', len(validset))

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    head_model = TT.get_model(args.model, patch_size=args.patch_size, num_channels=1 if args.grayscale else 3).cuda()
    TT.restore_snapshot(head_model, None, auto_file('linknet34_checkpoint.pth'))

    # Freeze model training
    for param in head_model.parameters():
        param.requires_grad = False

    afterburner = Afterburner()
    model = nn.Sequential(head_model, nn.Sigmoid(), afterburner).cuda()
    optimizer = TT.get_optimizer(args.optimizer, afterburner.parameters(), args.learning_rate)

    loss = TT.get_loss(args.loss).cuda()
    metrics = {'iou': JaccardScore().cuda(), 'accuracy': PixelAccuracy().cuda()}

    start_epoch = 0
    best_loss = np.inf
    train_history = pd.DataFrame()

    checkpoint_filename = os.path.join(experiment_dir, f'{args.model}_checkpoint.pth')
    if args.resume:
        start_epoch, train_history, best_loss = restore_snapshot(model, optimizer, checkpoint_filename)
        print('Resuming training from epoch', start_epoch, ' and loss', best_loss)
        print(train_history)

    print('Head       :', count_parameters(head_model))
    print('Afterburner:', count_parameters(afterburner))

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_scores = train(model, loss, optimizer, trainloader, epoch, metrics, summary_writer=writer)
        valid_loss, valid_scores = validate(model, loss, validloader, epoch, metrics, summary_writer=writer)

        summary = {
            'epoch': [epoch],
            'loss': [train_loss.avg],
            'val_loss': [valid_loss.avg]
        }

        for key, value in train_scores.items():
            summary[key] = [value.avg]

        for key, value in valid_scores.items():
            summary['val_' + key] = [value.avg]

        train_history = train_history.append(pd.DataFrame.from_dict(summary), ignore_index=True)

        print(epoch, summary)

        if valid_loss.avg < best_loss:
            save_snapshot(model, optimizer, valid_loss.avg, epoch, train_history, checkpoint_filename)
            best_loss = valid_loss.avg
            print('Checkpoint saved', epoch, best_loss)

    print('Training is finished...')

    train_history.to_csv(os.path.join(experiment_dir, args.experiment + '.csv'),
                         index=False,
                         mode='a' if args.resume else 'w',
                         header=not args.resume)


if __name__ == '__main__':
    main()
