import os


def set_gpus(gpus):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
