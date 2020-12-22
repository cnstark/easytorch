import os

import torch


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


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
