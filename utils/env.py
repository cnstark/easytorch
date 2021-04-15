import os

import torch


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def set_tf32_mode(tf32_mode):
    if torch.__version__ >= '1.7.0':
        if tf32_mode:
            print('Enable TF32 mode')
        else:
            # disable tf32 mode on A100
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print('Disable TF32 mode')
    else:
        if tf32_mode:
            raise RuntimeError('Torch version {} does not support tf32'.format(torch.__version__))
