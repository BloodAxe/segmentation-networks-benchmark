import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lib.train_utils import find_optimal_lr, auto_file
import torch_train as TT

if __name__ == '__main__':
    dd = 'e:/datasets/inria/train'

    model = TT.get_model('linknet34', patch_size=512, num_channels=3).cuda()
    loss = TT.get_loss('bce').cuda()
    optimizer = TT.get_optimizer('sgd', model.parameters(), 1e-4)
    trainset, validset, num_classes = TT.get_dataset('inria', dd, grayscale=False, patch_size=512)

    TT.restore_snapshot(model, None, auto_file('linknet34_checkpoint.pth'))
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    lr, loss = find_optimal_lr(model, loss, optimizer, trainloader)

    loss = np.convolve(loss, np.ones((4,)) / 4, mode='same')

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(lr, loss)
    ax.set(xlabel='lr', ylabel='loss', title='LR')
    ax.set_xscale("log", nonposx='clip')
    ax.grid()
    fig.show()

    plt.savefig('loss_plot.png')
    print(lr, loss)
    print('A')
