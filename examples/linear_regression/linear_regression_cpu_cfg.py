import os
from easytorch import Config

from linear_regression_runner import LinearRegressionRunner

CFG = Config()

CFG.DESC = 'linear_regression'
CFG.RUNNER = LinearRegressionRunner
CFG.DEVICE = 'cpu'

CFG.MODEL = Config()
CFG.MODEL.NAME = 'linear'

CFG.TRAIN = Config()

CFG.TRAIN.NUM_EPOCHS = 10000
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = Config()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.001,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = Config()
CFG.TRAIN.DATA.BATCH_SIZE = 10
CFG.TRAIN.DATA.K = 10
CFG.TRAIN.DATA.B = 6
CFG.TRAIN.DATA.NUM = 100
CFG.TRAIN.DATA.SHUFFLE = True
