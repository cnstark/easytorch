import os
import random
from typing import Dict

import torch
import numpy as np

from .logging import get_logger
from .dist import get_rank
from ..device import get_device_type, set_device_manual_seed


def set_visible_devices(devices: str):
    """Set environment variable `CUDA_VISIBLE_DEVICES` to select GPU devices.

    Examples:
        set_devices('0,1,2,3')

    Args:
        devices (str): environment variable `CUDA_VISIBLE_DEVICES` value
    """

    logger = get_logger('easytorch-env')
    if devices is not None:
        os.environ[{
            'gpu': 'CUDA_VISIBLE_DEVICES',
            'mlu': 'MLU_VISIBLE_DEVICES'
        }[get_device_type()]] = devices
        logger.info('Use devices {}.'.format(devices))
    else:
        logger.info('Use all devices.')


def set_tf32_mode(tf32_mode: bool):
    """Set tf32 mode on Ampere gpu when torch version >= 1.7.0 and cuda version >= 11.0.
    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere

    Args:
        tf32_mode (bool): set to ``True`` to enable tf32 mode.
    """

    logger = get_logger('easytorch-env')
    if get_device_type() == 'gpu':
        if torch.__version__ >= '1.7.0':
            if tf32_mode:
                logger.info('Enable TF32 mode')
            else:
                # disable tf32 mode on Ampere gpu
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logger.info('Disable TF32 mode')
        else:
            if tf32_mode:
                raise RuntimeError('Torch version {} does not support tf32'.format(torch.__version__))
    else:
        if tf32_mode:
            raise RuntimeError('Device {} does not support tf32.'.format(get_device_type()))


def setup_determinacy(seed: int, deterministic: bool = False, cudnn_enabled: bool = True,
                      cudnn_benchmark: bool = True, cudnn_deterministic: bool = False):
    """Setup determinacy.

    Including `python`, `random`, `numpy`, `torch`

    Args:
        seed (int): random seed.
        deterministic (bool): Use deterministic algorithms.
            See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html.
        cudnn_enabled (bool): Enable cudnn.
            See https://pytorch.org/docs/stable/backends.html
        cudnn_benchmark (bool): Enable cudnn benchmark.
            See https://pytorch.org/docs/stable/backends.html
        cudnn_deterministic (bool): Enable cudnn deterministic algorithms.
            See https://pytorch.org/docs/stable/backends.html
    """

    logger = get_logger('easytorch-env')

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    set_device_manual_seed(seed)

    if deterministic:
        if get_device_type() == 'gpu':
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        if torch.__version__ < '1.7.0':
            pass
        elif torch.__version__ < '1.8.0':
            torch.set_deterministic(True)
        else:
            torch.use_deterministic_algorithms(True)
        logger.info('Use deterministic algorithms.')

    if get_device_type() == 'gpu':
        if not cudnn_enabled:
            torch.backends.cudnn.enabled = False
            logger.info('Unset cudnn enabled.')
        if not cudnn_benchmark:
            torch.backends.cudnn.benchmark = False
            logger.info('Unset cudnn benchmark.')
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            logger.info('Set cudnn deterministic.')


def set_env(env_cfg: Dict):
    """Setup runtime env, include tf32, seed and determinacy.

    env config template:
    ```
    CFG.ENV = Config()
    CFG.ENV.TF32 = False
    CFG.ENV.SEED = 42
    CFG.ENV.DETERMINISTIC = True
    CFG.ENV.CUDNN = Config()
    CFG.ENV.CUDNN.ENABLED = False
    CFG.ENV.CUDNN.BENCHMARK = False
    CFG.ENV.CUDNN.DETERMINISTIC = True
    ```

    Args:
        env_cfg (Dict): env config.
    """

    # tf32
    set_tf32_mode(env_cfg.get('TF32', False))

    # determinacy
    seed = env_cfg.get('SEED')
    if seed is not None:
        # each rank has different seed in distributed mode
        setup_determinacy(
            seed + get_rank(),
            env_cfg.get('DETERMINISTIC', False),
            env_cfg.get('CUDNN.ENABLED', True),
            env_cfg.get('CUDNN.BENCHMARK', True),
            env_cfg.get('CUDNN.DETERMINISTIC', False)
        )
