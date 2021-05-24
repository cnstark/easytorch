import os
import glob
from logging import Logger

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import get_logger, get_rank


def get_ckpt_dict(model: nn.Module, optim: optim.Optimizer, epoch: int):
    if isinstance(model, DDP):
        _model = model.module
    else:
        _model = model
    return {
        'epoch': epoch,
        'model_state_dict': _model.state_dict(),
        'optim_state_dict': optim.state_dict()
    }


def get_last_ckpt_path(ckpt_save_dir: str, name_pattern: str='*.pt'):
    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    ckpt_list.sort()
    return ckpt_list[-1]


def load_ckpt(ckpt_save_dir: str, ckpt_path: str=None, logger: Logger=get_logger('easytorch')):
    if ckpt_path is None:
        ckpt_path = get_last_ckpt_path(ckpt_save_dir)
    logger.info('load ckpt from \'{}\''.format(ckpt_path))
    return torch.load(ckpt_path, map_location='cuda:{}'.format(get_rank()))


def save_ckpt(ckpt: dict, ckpt_path: str, logger: Logger=get_logger('easytorch')):
    torch.save(ckpt, ckpt_path)
    logger.info('ckpt {} saved'.format(ckpt_path))


def need_to_remove_last_ckpt(last_epoch: int, ckpt_save_strategy: int or list or tuple):
    if ckpt_save_strategy is None:
        return True
    elif isinstance(ckpt_save_strategy, int) and last_epoch % ckpt_save_strategy != 0:
        return True
    elif isinstance(ckpt_save_strategy, (list, tuple)) and not last_epoch in ckpt_save_strategy:
        return True
    else:
        return False


def backup_last_ckpt(last_ckpt_path: str, epoch: int, ckpt_save_strategy: int or list or tuple):
    last_epoch = epoch - 1

    # rename last ckpt to .bak
    if need_to_remove_last_ckpt(last_epoch, ckpt_save_strategy) and last_epoch != 0:
        os.rename(last_ckpt_path, last_ckpt_path + '.bak')


def clear_ckpt(ckpt_save_dir: str, name_pattern: str='*.pt.bak'):
    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    for ckpt in ckpt_list:
        os.remove(ckpt)
