import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


def plot_train_history(names, loss, val_loss, title=None, legend_loc='upper right'):
    fig = plt.figure(figsize=(15, 8))

    if title is not None:
        fig.suptitle(title)

    ax1, ax2 = fig.subplots(1, 2)

    for m in loss:
        ax1.plot(m)

    ax1.set_ylabel('Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Train')
    ax1.legend(names, loc=legend_loc)

    for m in val_loss:
        ax2.plot(m)

    ax2.set_ylabel('Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Test')
    ax2.legend(names, loc=legend_loc)

    plt.show()


def plot_experiment_train_history(names, loss, val_loss, title=None):
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, y=1.08)

    ax1, ax2 = fig.subplots(1, 2, sharey='all')

    for m in loss:
        ax1.plot(m)

    ax1.ylabel('Value')
    ax1.xlabel('Epoch')
    ax1.legend(names, loc='upper left')

    for m in val_loss:
        ax2.plot(m)

    ax2.ylabel('Value')
    ax2.xlabel('Epoch')
    ax2.legend(names, loc='upper left')

    plt.show()


def main():
    experiments = {
        'GCN': pd.read_csv(os.path.join('experiments', 'torch_gcn_224_rgb_bce', 'torch_gcn_224_rgb_bce.csv')),
        'Linknet + Resnet34': pd.read_csv(os.path.join('experiments', 'torch_linknet34_224_rgb_bce', 'torch_linknet34_224_rgb_bce.csv')),
        'Unet + VGG11': pd.read_csv(os.path.join('experiments', 'torch_unet11_224_rgb_bce', 'torch_unet11_224_rgb_bce.csv')),
    }

    names = []
    loss = []
    val_loss = []
    metric = []
    val_metric = []
    for key, item in experiments.items():
        names.append(key)
        loss.append(item[['loss']])
        val_loss.append(item[['val_loss']])

        metric.append(item[['jaccard']])
        val_metric.append(item[['val_jaccard']])

    plot_train_history(names, loss, val_loss, 'DSB2018, BCE loss', legend_loc='upper right')
    plot_train_history(names, metric, val_metric, 'DSB2018, Jaccard score', legend_loc='lower right')


if __name__ == '__main__':
    main()
