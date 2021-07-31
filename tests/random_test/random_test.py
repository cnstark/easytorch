import os
import sys
sys.path.append('../..')
import random
import numpy as np

from easydict import EasyDict
import torch
from torch import nn
from torch.utils.data import Dataset

from easytorch import Runner, get_rank, launch_training


class FakeDataset(Dataset):
    def __init__(self, num: int, min: int, max: int):
        self.num = num
        self.min = min
        self.max = max

    def __getitem__(self, index):
        return index, \
            random.randint(self.min, self.max), \
            np.random.randint(self.min, self.max + 1), \
            torch.randint(self.min, self.max + 1, (1,)).item()

    def __len__(self):
        return self.num


class DDPTestRunner(Runner):
    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return nn.Conv2d(3, 3, 3)

    @staticmethod
    def build_train_dataset(cfg: dict):
        return FakeDataset(cfg['TRAIN']['DATA']['NUM'], cfg['TRAIN']['DATA']['MIN'], cfg['TRAIN']['DATA']['MAX'])

    def train_iters(self, epoch, iter_index, data):
        print('rank: {:d}, epoch: {:d}, iter: {:d}, data: {}'.format(get_rank(), epoch, iter_index, data))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def build_cfg():
    CFG = EasyDict()

    CFG.DESC = 'ddp test'
    CFG.RUNNER = DDPTestRunner
    CFG.GPU_NUM = 8
    CFG.SEED = 6

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'conv'

    CFG.TRAIN = EasyDict()

    CFG.TRAIN.NUM_EPOCHS = 5
    CFG.TRAIN.CKPT_SAVE_DIR = 'checkpoints'

    CFG.TRAIN.CKPT_SAVE_STRATEGY = None

    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'SGD'
    CFG.TRAIN.OPTIM.PARAM = {
        'lr': 0.002,
        'momentum': 0.1,
    }

    CFG.TRAIN.DATA = EasyDict()
    CFG.TRAIN.DATA.NUM = 100
    CFG.TRAIN.DATA.MIN = 0
    CFG.TRAIN.DATA.MAX = 100
    CFG.TRAIN.DATA.BATCH_SIZE = 4
    CFG.TRAIN.DATA.NUM_WORKERS = 2
    CFG.TRAIN.DATA.SHUFFLE = True

    return CFG


if __name__ == "__main__":
    cfg = build_cfg()

    launch_training(cfg, gpus='0,1,2,3,4,5,6,7,8', tf32_mode=False)
