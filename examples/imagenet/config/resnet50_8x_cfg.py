import os
from easydict import EasyDict

from imagenet_runner import ImageNetRunner

CFG = EasyDict()

CFG.DESC = 'imagenet resnet50'
CFG.RUNNER = ImageNetRunner
CFG.GPU_NUM = 8

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'resnet50'

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 90
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'StepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'step_size': 30,
    'gamma': 0.1
}

IMAGENET_PATH = '/path/to/imagenet/jpegs'

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.SHUFFLE = True

CFG.TRAIN.DATA.DIR = os.path.join(IMAGENET_PATH, 'train')

CFG.VAL = EasyDict()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = os.path.join(IMAGENET_PATH, 'val')
