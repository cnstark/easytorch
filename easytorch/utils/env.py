import os
import random

import torch
import numpy as np

from .logging import get_logger


def set_gpus(gpus: str):
    """Set environment variable `CUDA_VISIBLE_DEVICES` to select GPU devices.

    Examples:
        set_gpus('0,1,2,3')

    Args:
        gpus (str): environment variable `CUDA_VISIBLE_DEVICES` value
    """

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def set_tf32_mode(tf32_mode: bool):
    """Set tf32 mode on Ampere gpu when torch version >= 1.7.0 and cuda version >= 11.0.
    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere

    Args:
        tf32_mode (bool): set to ``True`` to enable tf32 mode.
    """

    logger = get_logger('easytorch-env')
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


def setup_random_seed(seed: int):
    """Setup random seed.

    Including `random`, `numpy`, `torch`

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
