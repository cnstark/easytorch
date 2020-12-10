import os

import torch
from matplotlib import pyplot as plt


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def plot_per_epoch(save_dir, title, measurements, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    file_name = '{}.png'.format(title.replace(' ', '_').lower())
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path, dpi=200)
    plt.close()


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
