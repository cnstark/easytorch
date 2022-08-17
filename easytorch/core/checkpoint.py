import os
import re
import glob
from logging import Logger
from typing import Dict, List, Tuple, Union

import torch

from ..utils import get_logger, get_local_rank

DEFAULT_LOGGER = get_logger('easytorch-checkpoint')


def get_last_ckpt_path(ckpt_save_dir: str, name_pattern: str = r'^.+_[\d]*.pt$') -> str:
    r"""Get last checkpoint path in `ckpt_save_dir`
    checkpoint files will be sorted by name

    Args:
        ckpt_save_dir (str): checkpoint save directory
        name_pattern (str): re pattern for checkpoint file name, default is r'^.+_[\d]*.pt$'

    Returns:
        checkpoint path (str): last checkpoint path in `ckpt_save_dir`
    """

    ckpt_list = [f for f in os.listdir(ckpt_save_dir) if re.search(name_pattern, f) is not None]
    ckpt_list.sort()
    return os.path.join(ckpt_save_dir, ckpt_list[-1])


def load_ckpt(ckpt_save_dir: str, ckpt_path: str = None, use_gpu: bool = True,
              logger: Logger = DEFAULT_LOGGER) -> Dict:
    """Load checkpoint
    if param `ckpt_path` is None, load the last checkpoint in `ckpt_save_dir`,
    else load checkpoint from `ckpt_path`

    Args:
        ckpt_save_dir (str): checkpoint save directory
        ckpt_path (str): checkpoint path, default is None
        use_gpu (bool): set to ``True`` to load checkpoint to GPU
        logger (Logger): logger, default is Logger('easytorch')

    Returns:
        checkpoint dict loaded from file
    """

    if ckpt_path is None:
        ckpt_path = get_last_ckpt_path(ckpt_save_dir)
    if use_gpu:
        map_location = 'cuda:{}'.format(get_local_rank())
    else:
        map_location = 'cpu'
    logger.info('Loading Checkpoint from \'{}\''.format(ckpt_path))
    return torch.load(ckpt_path, map_location=map_location)


def save_ckpt(ckpt: Dict, ckpt_path: str, logger: Logger = DEFAULT_LOGGER):
    """Save checkpoint

    Args:
        ckpt (Dict): saved checkpoint dict
        ckpt_path (str): checkpoint save path
        logger (Logger): logger, default is Logger('easytorch')
    """

    torch.save(ckpt, ckpt_path)
    logger.info('Checkpoint {} saved'.format(ckpt_path))


def need_to_remove_last_ckpt(last_epoch: int, ckpt_save_strategy: Union[int, List, Tuple]) -> bool:
    """Judging whether to remove last checkpoint by `ckpt_save_strategy`

    `ckpt_save_strategy` should be None, an int value, a list or a tuple
    if `ckpt_save_strategy` is None, remove last checkpoint file every epoch
    if `ckpt_save_strategy` is an int value `n`, save checkpoint every n epoch,
        remove last checkpoint file when last_epoch % ckpt_save_strategy != 0
    if `ckpt_save_strategy` is a list or a tuple `l`, save checkpoint when epoch in `l`,
        remove last checkpoint file when last_epoch not in ckpt_save_strategy

    Args:
        last_epoch (int): last epoch num
        ckpt_save_strategy (Union[int, List, Tuple]): checkpoint save strategy

    Returns:
        last checkpoint delete flag (bool): `True` means delete last checkpoint
    """

    if ckpt_save_strategy is None:
        return True
    elif isinstance(ckpt_save_strategy, int) and last_epoch % ckpt_save_strategy != 0:
        return True
    elif isinstance(ckpt_save_strategy, (list, tuple)) and last_epoch not in ckpt_save_strategy:
        return True
    else:
        return False


def backup_last_ckpt(last_ckpt_path: str, epoch: int, ckpt_save_strategy: Union[int, List, Tuple]):
    """Backup last checkpoint when last checkpoint needs to be removed (by call need_to_remove_last_ckpt())
    if last checkpoint file name is `a.pt`, rename `a.pt` to `a.pt.bak`

    Args:
        last_ckpt_path (str): last checkpoint file path
        epoch (int): current epoch num
        ckpt_save_strategy (Union[int, List, Tuple]): checkpoint save strategy
    """

    last_epoch = epoch - 1

    # rename last ckpt to .bak
    if need_to_remove_last_ckpt(last_epoch, ckpt_save_strategy) and last_epoch != 0:
        os.rename(last_ckpt_path, last_ckpt_path + '.bak')


def clear_ckpt(ckpt_save_dir: str, name_pattern: str = '*.pt.bak'):
    """Clear all backed up checkpoint files

    Args:
        ckpt_save_dir (str): checkpoint save directory
        name_pattern (str): backed up checkpoint file name pattern
    """

    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    for ckpt in ckpt_list:
        os.remove(ckpt)
