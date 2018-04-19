import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_train_history(names, loss, val_loss, title=None):
    fig = plt.figure(figsize=(15, 8))

    if title is not None:
        fig.suptitle(title)

    plt.subplot(1, 2, 1)
    for m in loss:
        plt.plot(m)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(names, loc='upper left')

    plt.subplot(1, 2, 2)
    for m in val_loss:
        plt.plot(m)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(names, loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()



def plot_experiment_train_history(names, loss, val_loss, title=None):
    fig = plt.figure(figsize=(15, 8))

    if title is not None:
        fig.suptitle(title)

    plt.subplot(1, 2, 1)
    for m in loss:
        plt.plot(m)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(names, loc='upper left')

    plt.subplot(1, 2, 2)
    for m in val_loss:
        plt.plot(m)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(names, loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def main():
    experiments = {
        # 'zf_unet': pd.read_csv(os.path.join('zf_unet_512_rgb_bce', 'zf_unet_512_rgb_bce.csv')),
        'linknet': pd.read_csv(os.path.join('linknet_512_rgb_bce', 'linknet_512_rgb_bce.csv')),
        # 'dilated_unet': os.path.join('dilated_unet_512_rgb_bce', 'dilated_unet_512_rgb_bce.csv'),
        'dilated_resnet': pd.read_csv(os.path.join('dilated_resnet_512_rgb_bce', 'dilated_resnet_512_rgb_bce.csv')),
    }

    names = []
    loss = []
    val_loss = []
    for key, item in experiments.items():
        names.append(key)
        loss.append(item[['loss']])
        val_loss.append(item[['val_loss']])

    plot_train_history(names, loss, val_loss)


if __name__ == '__main__':
    main()
