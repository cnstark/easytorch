import os

import torch


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
