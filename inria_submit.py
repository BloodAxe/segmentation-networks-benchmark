import argparse
from multiprocessing.pool import Pool

import cv2
import os.path
import sys

from torch.backends import cudnn

from lib import augmentations as aug
import numpy as np
import pandas as pd
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import torch_train as TT
from lib.common import find_in_dir, read_rgb, InMemoryDataset
from lib.datasets.Inria import INRIA, INRIA_MEAN, INRIA_STD
from lib.datasets.dsb2018 import DSB2018Sliced
from lib.losses import JaccardLoss, FocalLossBinary, BCEWithLogitsLossAndSmoothJaccard, BCEWithSigmoidLoss
from lib.metrics import JaccardScore, PixelAccuracy
from lib.models import linknet, unet16, unet11
from lib.models.duc_hdc import ResNetDUCHDC, ResNetDUC
from lib.models.gcn152 import GCN152, GCN34
from lib.models.psp_net import PSPNet
from lib.models.tiramisu import FCDenseNet67
from lib.models.unet import UNet
from lib.models.zf_unet import ZF_UNET
from lib.tiles import ImageSlicer
from lib.train_utils import AverageMeter, auto_file

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def get_dataset(dataset_name, dataset_dir, grayscale, patch_size, keep_in_mem=False):
    dataset_name = dataset_name.lower()

    if dataset_name == 'inria':
        return INRIA(dataset_dir, grayscale, patch_size, keep_in_mem)

    if dataset_name == 'dsb2018':
        return DSB2018Sliced(dataset_dir, grayscale, patch_size)

    raise ValueError(dataset_name)


def preduct(model, loss, optimizer, dataloader, epoch: int, metrics={}, summary_writer=None):
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


def predict_full(image, model, test_transform):
    image, pad = aug.pad(image, 32, borderType=cv2.BORDER_REPLICATE)
    image, _ = test_transform(image)
    images = list(aug.tta_d4_aug([image]))
    predicts = []

    for image in images:
        image = torch.from_numpy(np.moveaxis(image, -1, 0)).float().unsqueeze(dim=0)
        image = image.cuda(non_blocking=True)
        y = model(image)
        y = torch.sigmoid(y).cpu().numpy()
        y = np.moveaxis(y, 1, -1)
        y = np.squeeze(y)
        predicts.append(y)

    mask = next(aug.tta_d4_deaug(predicts))
    mask = aug.unpad(mask, pad)
    return mask


def predict_tiled(image, model, test_transform, patch_size, batch_size):
    image, _ = test_transform(image)

    slicer = ImageSlicer(image.shape, patch_size, patch_size // 2, weight='pyramid')
    patches = slicer.split(image)

    patches = aug.tta_d4_aug(patches)
    testset = InMemoryDataset(patches, None)
    trainloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    patches_pred = []
    for batch_index, x in enumerate(trainloader):
        x = x.cuda(non_blocking=True)
        y = model(x)
        y = torch.sigmoid(y).cpu().numpy()
        y = np.moveaxis(y, 1, -1)
        patches_pred.extend(y)

    patches_pred = aug.tta_d4_deaug(patches_pred)
    mask = slicer.merge(patches_pred, dtype=np.float32)
    return mask


def main():
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--grayscale', action='store_true', help='Whether to use grayscale image instead of RGB')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of the model')
    parser.add_argument('-c', '--checkpoint', required=True, type=str, help='Name of the model checkpoint')
    parser.add_argument('-p', '--patch-size', type=int, default=224)
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Root directory where datasets are located.')
    parser.add_argument('-x', '--experiment', type=str, help='Name of the experiment')
    parser.add_argument('-f', '--full', action='store_true')

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = 'inria_%s_%d_%s' % (args.model, args.patch_size, 'gray' if args.grayscale else 'rgb')

    experiment_dir = os.path.join('submits', args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    model = TT.get_model(args.model, patch_size=args.patch_size, num_channels=1 if args.grayscale else 3).cuda()
    start_epoch, train_history, best_loss = TT.restore_snapshot(model, None, auto_file(args.checkpoint))
    print('Using weights from epoch', start_epoch - 1, best_loss)

    test_transform = aug.Sequential([
        aug.ImageOnly(aug.NormalizeImage(mean=INRIA_MEAN, std=INRIA_STD)),
    ])

    x = sorted(find_in_dir(os.path.join(args.data_dir, 'images')))
    # x = x[:10]

    model.eval()
    with torch.no_grad():

        for test_fname in tqdm(x, total=len(x)):
            image = read_rgb(test_fname)
            basename = os.path.splitext(os.path.basename(test_fname))[0]

            if args.full:
                mask = predict_full(image, model, test_transform)
            else:
                mask = predict_tiled(image, model, test_transform, args.patch_size, args.batch_size)

            mask = ((mask > 0.5) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(experiment_dir, basename + '.tif'), mask)


if __name__ == '__main__':
    main()
