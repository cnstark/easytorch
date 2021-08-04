import os
import random
from typing import Callable

import torch
from torch import distributed as dist
from torch.distributed import Backend
from torch import multiprocessing as mp

from ..config import import_config, config_md5, save_config, copy_config_file
from ..utils import set_gpus, set_tf32_mode


def train(cfg: dict, use_gpu: bool, tf32_mode: bool):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (dict): Easytorch config.
        use_gpu (bool):  set to ``True`` to use GPU
        tf32_mode (dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    if use_gpu:
        # set tf32 mode
        set_tf32_mode(tf32_mode)

    # init runner
    Runner = cfg['RUNNER']
    runner = Runner(cfg, use_gpu)

    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')

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
    train(cfg, True, tf32_mode)


def launch_training(cfg: dict or str, gpus: str, tf32_mode: bool):
    """Launch training process defined by `cfg`.

    Support distributed data parallel training when the number of available GPUs is greater than one.
    Nccl backend is used by default.

    Notes:
        If `USE_GPU` in `cfg` is ```True```, easytorch will run in GPU mode, `GPU_NUM` in `cfg` must
        be greater than 0;
        If `USE_GPU` in `cfg` is ```False```, easytorch will run in CPU mode, `GPU_NUM` in `cfg` must
        be 0.
        In order to ensure the consistency of training results, the number of available GPUs
        must be equal to `GPU_NUM` in GPU mode.

    Args:
        cfg (dict): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        tf32_mode(dict): set to ``True`` to use tf32 on Ampere GPU.
    """

    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg)
    else:
        cfg_path = None

    use_gpu = cfg.get('USE_GPU', True)
    gpu_num = cfg.get('GPU_NUM', 0)

    if use_gpu:
        if gpu_num == 0:
            raise RuntimeError('Easytorch is running in GPU mode, but cfg.GPU_NUM is 0')

        set_gpus(gpus)

        world_size = torch.cuda.device_count()
        if gpu_num != world_size:
            raise RuntimeError('GPU num not match, cfg.GPU_NUM = {:d}, but torch.cuda.device_count() = {:d}'.format(
                gpu_num, world_size
            ))
    else:
        if gpu_num != 0:
            raise RuntimeError('Easytorch is running in CPU mode, but cfg.GPU_NUM is not zero')

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
        train(cfg, use_gpu, tf32_mode)
    else:
        # default backend
        backend = Backend.NCCL

        # random port
        port = random.randint(50000, 65000)
        init_method = 'tcp://127.0.0.1:{:d}'.format(port)

        mp.spawn(
            train_ddp,
            args=(gpu_num, backend, init_method, cfg, tf32_mode),
            nprocs=gpu_num,
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

    use_gpu = cfg.get('USE_GPU', True)
    if use_gpu:
        set_gpus(gpus)
        set_tf32_mode(tf32_mode)

    # convert ckpt save dir
    cfg['TRAIN']['CKPT_SAVE_DIR'] = os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], config_md5(cfg))

    # make ckpt save dir
    if not os.path.isdir(cfg['TRAIN']['CKPT_SAVE_DIR']):
        os.makedirs(cfg['TRAIN']['CKPT_SAVE_DIR'])

    Runner = cfg['RUNNER']
    runner = Runner(cfg, use_gpu)

    fn(cfg, runner, *args)
