import os
from easydict import EasyDict

from mnist_runner import MNISTRunner

CFG = EasyDict()

CFG.DESC = 'mnist'
CFG.RUNNER = MNISTRunner
CFG.USE_GPU = False

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'conv_net'

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 30
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.002,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.DIR = 'mnist_data'
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = 'mnist_data'
