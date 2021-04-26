import os

import torch

from .logging import get_logger


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def set_tf32_mode(tf32_mode):
    logger = get_logger('easytorch-env')
    if torch.__version__ >= '1.7.0':
        if tf32_mode:
            logger.info('Enable TF32 mode')
        else:
            # disable tf32 mode on A100
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info('Disable TF32 mode')
    else:
        if tf32_mode:
            raise RuntimeError('Torch version {} does not support tf32'.format(torch.__version__))
