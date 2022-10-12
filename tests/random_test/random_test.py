import random
from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from easytorch import Config, Runner, get_rank, launch_training


class FakeDataset(Dataset):
    """FakeDataset
    """

    def __init__(self, num: int, min_: int, max_: int):
        self.num = num
        self.min = min_
        self.max = max_

    def __getitem__(self, index):
        return index, \
            random.randint(self.min, self.max), \
            np.random.randint(self.min, self.max + 1), \
            torch.randint(self.min, self.max + 1, (1,)).item()

    def __len__(self):
        return self.num


class DDPTestRunner(Runner):
    """DDPTestRunner
    """

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return nn.Conv2d(3, 3, 3)

    @staticmethod
    def build_train_dataset(cfg: Dict):
        return FakeDataset(cfg['TRAIN']['DATA']['NUM'], cfg['TRAIN']['DATA']['MIN'], cfg['TRAIN']['DATA']['MAX'])

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        print('rank: {:d}, epoch: {:d}, iter: {:d}, data: {}'.format(get_rank(), epoch, iter_index, data))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def build_cfg():
    CFG = Config()

    CFG.DESC = 'ddp test'
    CFG.RUNNER = DDPTestRunner
    CFG.GPU_NUM = 8

    CFG.ENV = Config()
    CFG.ENV.TF32 = False
    CFG.ENV.SEED = 6

    CFG.MODEL = Config()
    CFG.MODEL.NAME = 'conv'

    CFG.TRAIN = Config()

    CFG.TRAIN.NUM_EPOCHS = 5
    CFG.TRAIN.CKPT_SAVE_DIR = 'checkpoints'

    CFG.TRAIN.CKPT_SAVE_STRATEGY = None

    CFG.TRAIN.OPTIM = Config()
    CFG.TRAIN.OPTIM.TYPE = 'SGD'
    CFG.TRAIN.OPTIM.PARAM = {
        'lr': 0.002,
        'momentum': 0.1,
    }

    CFG.TRAIN.DATA = Config()
    CFG.TRAIN.DATA.NUM = 100
    CFG.TRAIN.DATA.MIN = 0
    CFG.TRAIN.DATA.MAX = 100
    CFG.TRAIN.DATA.BATCH_SIZE = 4
    CFG.TRAIN.DATA.NUM_WORKERS = 2
    CFG.TRAIN.DATA.SHUFFLE = True

    return CFG


if __name__ == '__main__':
    cfg_ = build_cfg()

    launch_training(cfg_, devices='0,1,2,3,4,5,6,7,8')
