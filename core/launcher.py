import time
import random
from typing import Callable

import torch
from torch import distributed as dist
from torch.distributed import Backend
from torch import multiprocessing as mp

from ..config import import_config
from ..utils import set_gpus, set_tf32_mode


def train(cfg: dict, tf32_mode: bool):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (dict): Easytorch config.
        tf32_mode (dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    # set tf32 mode
    set_tf32_mode(tf32_mode)

    # init runner
    Runner = cfg['RUNNER']
    runner = Runner(cfg)

    # train
    runner.train(cfg)


def train_ddp(rank: int, world_size: int, backend: str or Backend, init_method: str, cfg: dict, tf32_mode: bool):
    """Start training with DistributedDataParallel

    Args:
        rank: Rank of the current process.
        world_size: Number of processes participating in the job.
        backend: The backend to use.
        init_method: URL specifying how to initialize the process group.
        cfg (dict): Easytorch config.
        tf32_mode (dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    # set cuda device
    torch.cuda.set_device(rank)

    # init process
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    # start training
    train(cfg, tf32_mode)


def launch_training(cfg: dict or str, gpus: str, tf32_mode: bool):
    """Launch training process defined by `cfg`.

    Support distributed data parallel training when the number of available GPUs is greater than one.
    Nccl backend is used by default.

    Note:
        In order to ensure the consistency of training results, the number of available GPUs
        must be equal to `GPU_NUM` in `cfg`

    Args:
        cfg (dict): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        tf32_mode(dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    if isinstance(cfg, str):
        cfg = import_config(cfg)

    set_gpus(gpus)

    world_size = torch.cuda.device_count()

    if cfg.GPU_NUM != world_size:
        raise RuntimeError('GPU num not match, cfg.GPU_NUM = {:d}, but torch.cuda.device_count() = {:d}'.format(
            cfg.GPU_NUM, world_size
        ))

    if world_size == 0:
        raise RuntimeError('No available gpus')
    elif world_size == 1:
        train(cfg, tf32_mode)
    else:
        # default backend
        backend = Backend.NCCL

        # random port
        port = random.randint(50000, 65000)
        init_method = 'tcp://127.0.0.1:{:d}'.format(port)

        mp.spawn(
            train_ddp,
            args=(world_size, backend, init_method, cfg, tf32_mode),
            nprocs=world_size,
            join=True
        )


def launch_runner(cfg: dict or str, fn: Callable, args: tuple = (), gpus: str = None, tf32_mode: bool = False):
    """Launch runner defined by `cfg`, and call `fn`.

    Args:
        cfg (dict): Easytorch config.
        fn (Callable): Function is called after init runner.
            The function is called as ``fn(cfg, runner, *args)``, where ``cfg`` is
            the Easytorch config and ``runner`` is the runner defined by ``cfg`` and
            ``args`` is the passed through tuple of arguments.
        args (tuple): Arguments passed to ``fn``.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        tf32_mode(dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    if isinstance(cfg, str):
        cfg = import_config(cfg)

    set_gpus(gpus)
    set_tf32_mode(tf32_mode)

    Runner = cfg['RUNNER']
    runner = Runner(cfg)

    fn(cfg, runner, *args)
