import random
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config import import_config
from ..utils import set_gpus, set_tf32_mode


def train(cfg: dict, tf32_mode: str):
    set_tf32_mode(tf32_mode)

    Runner = cfg.RUNNER
    runner = Runner(cfg)

    runner.train(cfg)


def init_process(rank: int, world_size: int, nccl_port: int):
    dist.init_process_group(
        'nccl',
        init_method='tcp://127.0.0.1:{:d}'.format(nccl_port),
        rank=rank,
        world_size=world_size
    )


def train_ddp(
        rank: int, world_size: int, nccl_port: int,
        cfg: dict, tf32_mode: str
    ):
    init_process(rank, world_size, nccl_port)

    torch.cuda.set_device(rank)

    set_tf32_mode(tf32_mode)

    Runner = cfg.RUNNER
    runner = Runner(cfg)

    runner.train(cfg)


def launch_training(cfg_path: str, gpus: str, tf32_mode: bool):
    cfg = import_config(cfg_path)
    set_gpus(gpus)

    world_size = torch.cuda.device_count()

    if cfg.GPU_NUM != world_size:
        raise RuntimeError(
            ('GPU num not match, cfg.GPU_NUM = {:d}, ' +
            'but torch.cuda.device_count() = {:d}').format(cfg.GPU_NUM, world_size)
        )

    if world_size == 0:
        raise RuntimeError('No available gpus')
    elif world_size == 1:
        train(cfg, tf32_mode)
    else:
        nccl_port = random.randint(50000, 65000)
        mp.spawn(
            train_ddp,
            args=(world_size, nccl_port, cfg, tf32_mode),
            nprocs=world_size,
            join=True
        )


def launch_inference_runner(cfg_path: str, gpus: str, tf32_mode: bool):
    cfg = import_config(cfg_path)
    set_gpus(gpus)
    set_tf32_mode(tf32_mode)

    Runner = cfg.RUNNER
    runner = Runner(cfg)

    runner.model.eval()

    return cfg, runner
