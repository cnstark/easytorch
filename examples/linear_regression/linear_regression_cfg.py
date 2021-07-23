import os
from easydict import EasyDict

from linear_regression_runner import LinearRegressionRunner

CFG = EasyDict()

CFG.DESC = 'linear_regression'
CFG.RUNNER = LinearRegressionRunner
CFG.USE_GPU = False

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'linear'

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 10000
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.001,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 10
CFG.TRAIN.DATA.K = 10
CFG.TRAIN.DATA.B = 6
CFG.TRAIN.DATA.NUM = 100
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.K = 10
CFG.VAL.DATA.B = 6
CFG.VAL.DATA.NUM = 100
