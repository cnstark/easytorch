import os
from easytorch import Config

from mnist_runner import MNISTRunner

CFG = Config()

CFG.DESC = 'mnist'
CFG.RUNNER = MNISTRunner
CFG.DEVICE = 'cpu'

CFG.MODEL = Config()
CFG.MODEL.NAME = 'conv_net'

CFG.TRAIN = Config()

CFG.TRAIN.NUM_EPOCHS = 30
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = Config()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.002,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = Config()
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.DIR = 'mnist_data'
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = Config()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = Config()
CFG.VAL.DATA.DIR = 'mnist_data'
