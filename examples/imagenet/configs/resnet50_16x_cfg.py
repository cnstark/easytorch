import os
from easytorch import Config

from imagenet_runner import ImagenetRunner

CFG = Config()

CFG.DESC = 'imagenet resnet50'
CFG.RUNNER = ImagenetRunner
CFG.DEVICE = 'gpu'
CFG.DEVICE_NUM = 8
CFG.DIST_NODE_NUM = 2
CFG.DIST_BACKEND = 'nccl'
CFG.DIST_INIT_METHOD='tcp://{ip_of_node_0}:{free_port}'

CFG.MODEL = Config()
CFG.MODEL.NAME = 'resnet50'

CFG.TRAIN = Config()

CFG.TRAIN.NUM_EPOCHS = 90
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = Config()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4
}

CFG.TRAIN.LR_SCHEDULER = Config()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'StepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'step_size': 30,
    'gamma': 0.1
}

IMAGENET_PATH = 'datasets/imagenet/jpegs'

CFG.TRAIN.DATA = Config()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.SHUFFLE = True

CFG.TRAIN.DATA.DIR = os.path.join(IMAGENET_PATH, 'train')
CFG.TRAIN.DATA.CROP_SIZE = 224
CFG.TRAIN.DATA.NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CFG.VAL = Config()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = Config()
CFG.VAL.DATA.BATCH_SIZE = 16
CFG.VAL.DATA.DIR = os.path.join(IMAGENET_PATH, 'val')
CFG.VAL.DATA.CROP_SIZE = 224
CFG.VAL.DATA.RESIZE = 256
CFG.VAL.DATA.NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
