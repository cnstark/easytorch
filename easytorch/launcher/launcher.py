import os
from typing import Callable, Dict, Union, Tuple

import torch

from ..config import import_config, save_config, copy_config_file, convert_config
from ..utils import set_gpus
from .dist import dist_wrap
from .train import train


def init_cfg(cfg: Union[Dict, str], save: bool = False):
    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg)
    else:
        cfg_path = None

    # convert ckpt save dir
    convert_config(cfg)

    # save config
    if save and not os.path.isdir(cfg['TRAIN']['CKPT_SAVE_DIR']):
        os.makedirs(cfg['TRAIN']['CKPT_SAVE_DIR'])
        save_config(cfg, os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], 'cfg.txt'))
        if cfg_path is not None:
            copy_config_file(cfg_path, cfg['TRAIN']['CKPT_SAVE_DIR'])

    return cfg


def launch_training(cfg: Union[Dict, str], gpus: str, node_rank: int = 0):
    """Launch training process defined by `cfg`.

    Support distributed data parallel training when the number of available GPUs is greater than one.
    Nccl backend is used by default.

    Notes:
        If `GPU_NUM` in `cfg` is greater than `0`, easytorch will run in GPU mode;
        If `GPU_NUM` in `cfg` is `0`, easytorch will run in CPU mode.
        In order to ensure the consistency of training results, the number of available GPUs
        must be equal to `GPU_NUM` in GPU mode.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    cfg = init_cfg(cfg, node_rank == 0)

    gpu_num = cfg.get('GPU_NUM', 0)

    if gpu_num != 0:
        set_gpus(gpus)

        device_count = torch.cuda.device_count()
        if gpu_num != device_count:
            raise RuntimeError('GPU num not match, cfg.GPU_NUM = {:d}, but torch.cuda.device_count() = {:d}'.format(
                gpu_num, device_count
            ))

    train_dist = dist_wrap(
        train,
        node_num=cfg.get('DIST_NODE_NUM', 1),
        gpu_num=gpu_num,
        node_rank=node_rank,
        dist_backend=cfg.get('DIST_BACKEND'),
        init_method=cfg.get('DIST_INIT_METHOD')
    )

    train_dist(cfg)


def launch_runner(cfg: Union[Dict, str], fn: Callable, args: Tuple = (), gpus: str = None):
    """Launch runner defined by `cfg`, and call `fn`.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        fn (Callable): Function is called after init runner.
            The function is called as ``fn(cfg, runner, *args)``, where ``cfg`` is
            the Easytorch config and ``runner`` is the runner defined by ``cfg`` and
            ``args`` is the passed through tuple of arguments.
        args (tuple): Arguments passed to ``fn``.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
    """

    cfg = init_cfg(cfg, True)

    if cfg.get('GPU_NUM', 0) != 0:
        set_gpus(gpus)

    Runner = cfg['RUNNER']
    runner = Runner(cfg)

    fn(cfg, runner, *args)
