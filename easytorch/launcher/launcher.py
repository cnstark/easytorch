import os
import traceback
from typing import Callable, Dict, Union, Tuple

from ..config import import_config, save_config_str, copy_config_file, convert_config, get_ckpt_save_dir
from ..utils import set_gpus, get_logger
from .dist_wrap import dist_wrap


def init_cfg(cfg: Union[Dict, str], save: bool = False):
    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg, verbose=save)
    else:
        cfg_path = None

    # convert ckpt save dir
    convert_config(cfg)

    # save config
    ckpt_save_dir = get_ckpt_save_dir(cfg)
    if save and not os.path.isdir(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
        save_config_str(cfg, os.path.join(ckpt_save_dir, 'cfg.txt'))
        if cfg_path is not None:
            copy_config_file(cfg_path, ckpt_save_dir)

    return cfg


def training_func(cfg: Dict):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (Dict): Easytorch config.
    """

    # init runner
    logger = get_logger('easytorch-launcher')
    logger.info('Initializing runner "{}"'.format(cfg['RUNNER']))
    runner = cfg['RUNNER'](cfg)

    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')

    # train
    try:
        runner.train(cfg)
    except BaseException as e:
        # log exception to file
        runner.logger.error(traceback.format_exc())
        raise e


def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
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

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch training.')

    cfg = init_cfg(cfg, node_rank == 0)

    if cfg.get('GPU_NUM', 0) != 0:
        set_gpus(gpus)

    train_dist = dist_wrap(
        training_func,
        node_num=cfg.get('DIST_NODE_NUM', 1),
        gpu_num=cfg.get('GPU_NUM', 0),
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

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch runner.')

    cfg = init_cfg(cfg, True)

    if cfg.get('GPU_NUM', 0) != 0:
        set_gpus(gpus)

    # init runner
    runner = cfg['RUNNER'](cfg)

    # call fn
    fn(cfg, runner, *args)
