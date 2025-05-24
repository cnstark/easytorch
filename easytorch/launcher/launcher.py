import traceback
from typing import Callable, Dict, Union, Tuple

from ..config import init_cfg
from ..utils import set_visible_devices, get_logger, is_master
from ..device import set_device_type
from .dist_wrap import dist_wrap

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
    if is_master():
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


def launch_training(cfg: Union[Dict, str], devices: str = None, node_rank: int = 0):
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
        devices (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch training.')

    cfg = init_cfg(cfg, node_rank == 0)

    if cfg.get('DEVICE') is not None:
        set_device_type(cfg['DEVICE'])
        device_num = cfg.get('DEVICE_NUM', 0)
    elif cfg.get('GPU_NUM', 0) != 0 or cfg.get('MLU_NUM', 0) != 0:
        if cfg.get('GPU_NUM', 0) != 0 and cfg.get('MLU_NUM', 0) == 0:
            set_device_type('gpu')
            device_num = cfg.get('GPU_NUM', 0)
        elif cfg.get('GPU_NUM', 0) == 0 and cfg.get('MLU_NUM', 0) != 0:
            set_device_type('mlu')
            device_num = cfg.get('MLU_NUM', 0)
        else:
            raise ValueError('At least one of `CFG.GPU_NUM` and `CFG.MLU_NUM` is 0.')
        set_visible_devices(devices)
    else:
        set_device_type('cpu')
        device_num = 0

    train_dist = dist_wrap(
        training_func,
        node_num=cfg.get('DIST_NODE_NUM', 1),
        device_num=device_num,
        node_rank=node_rank,
        dist_backend=cfg.get('DIST_BACKEND'),
        init_method=cfg.get('DIST_INIT_METHOD')
    )
    train_dist(cfg)


def launch_runner(cfg: Union[Dict, str], fn: Callable, args: Tuple = (), device_type: str = 'gpu', devices: str = None):
    """Launch runner defined by `cfg`, and call `fn`.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        fn (Callable): Function is called after init runner.
            The function is called as ``fn(cfg, runner, *args)``, where ``cfg`` is
            the Easytorch config and ``runner`` is the runner defined by ``cfg`` and
            ``args`` is the passed through tuple of arguments.
        args (tuple): Arguments passed to ``fn``.
        device_type (str): Device type. Valid values are ['cpu', 'gpu', 'mlu'].
        devices (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
    """

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch runner.')

    cfg = init_cfg(cfg, True)

    set_device_type(device_type)

    if device_type != 'cpu':
        set_visible_devices(devices)

    # init runner
    runner = cfg['RUNNER'](cfg)

    # call fn
    return fn(cfg, runner, *args)
