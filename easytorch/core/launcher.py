import os
from typing import Callable

import torch
from torch import distributed as dist
from torch.distributed import Backend
from torch import multiprocessing as mp

from ..config import import_config, config_md5, save_config, copy_config_file
from ..utils import set_gpus, get_dist_backend


def train(cfg: dict):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (dict): Easytorch config.
    """

    # init runner
    Runner = cfg['RUNNER']
    runner = Runner(cfg)

    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')

    # train
    runner.train(cfg)


def train_ddp(local_rank: int, world_size: int, backend: str or Backend, init_method: str, cfg: dict,
              node_rank: int = 0):
    """Start training with DistributedDataParallel

    Args:
        local_rank: Rank of the current process in the current node.
        world_size: Number of processes participating in the job.
        backend: The backend to use.
        init_method: URL specifying how to initialize the process group.
        cfg (dict): Easytorch config.
        node_rank (int): Rank of the current node.
    """

    # set cuda device
    torch.cuda.set_device(local_rank)

    rank = cfg['GPU_NUM'] * node_rank + local_rank

    # init process
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    # start training
    train(cfg)


def launch_training(cfg: dict or str, gpus: str, node_rank: int = 0):
    """Launch training process defined by `cfg`.

    Support distributed data parallel training when the number of available GPUs is greater than one.
    Nccl backend is used by default.

    Notes:
        If `GPU_NUM` in `cfg` is greater than `0`, easytorch will run in GPU mode;
        If `GPU_NUM` in `cfg` is `0`, easytorch will run in CPU mode.
        In order to ensure the consistency of training results, the number of available GPUs
        must be equal to `GPU_NUM` in GPU mode.

    Args:
        cfg (dict): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg)
    else:
        cfg_path = None

    gpu_num = cfg.get('GPU_NUM', 0)

    if gpu_num != 0:
        set_gpus(gpus)

        device_count = torch.cuda.device_count()
        if gpu_num != device_count:
            raise RuntimeError('GPU num not match, cfg.GPU_NUM = {:d}, but torch.cuda.device_count() = {:d}'.format(
                gpu_num, device_count
            ))

    # convert ckpt save dir
    cfg['TRAIN']['CKPT_SAVE_DIR'] = os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], config_md5(cfg))

    # save config
    if not os.path.isdir(cfg['TRAIN']['CKPT_SAVE_DIR']):
        os.makedirs(cfg['TRAIN']['CKPT_SAVE_DIR'])
        if cfg_path is None:
            save_config(cfg, os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], 'param.txt'))
        else:
            copy_config_file(cfg_path, cfg['TRAIN']['CKPT_SAVE_DIR'])

    if gpu_num <= 1:
        train(cfg)
    else:
        dist_node_num = cfg.get('DIST_NODE_NUM', 1)
        if node_rank >= dist_node_num:
            raise ValueError('The node_rank must be less than dist_node_num!')

        world_size = dist_node_num * gpu_num

        backend, init_method = get_dist_backend(dist_node_num, cfg.get('DIST_BACKEND'), cfg.get('DIST_INIT_METHOD'))

        mp.spawn(
            train_ddp,
            args=(world_size, backend, init_method, cfg, node_rank),
            nprocs=gpu_num,
            join=True
        )


def launch_runner(cfg: dict or str, fn: Callable, args: tuple = (), gpus: str = None):
    """Launch runner defined by `cfg`, and call `fn`.

    Args:
        cfg (dict): Easytorch config.
        fn (Callable): Function is called after init runner.
            The function is called as ``fn(cfg, runner, *args)``, where ``cfg`` is
            the Easytorch config and ``runner`` is the runner defined by ``cfg`` and
            ``args`` is the passed through tuple of arguments.
        args (tuple): Arguments passed to ``fn``.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
    """

    if isinstance(cfg, str):
        cfg = import_config(cfg)

    if cfg.get('GPU_NUM', 0) != 0:
        set_gpus(gpus)

    # convert ckpt save dir
    cfg['TRAIN']['CKPT_SAVE_DIR'] = os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], config_md5(cfg))

    # make ckpt save dir
    if not os.path.isdir(cfg['TRAIN']['CKPT_SAVE_DIR']):
        os.makedirs(cfg['TRAIN']['CKPT_SAVE_DIR'])

    Runner = cfg['RUNNER']
    runner = Runner(cfg)

    fn(cfg, runner, *args)
