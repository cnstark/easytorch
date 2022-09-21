"""Everything is based on config.

`Config` is the set of all configurations. `Config` is is implemented by `dict`, We recommend using `Config`.

Look at the following example:

cfg.py

```python
import os
from easytorch import Config

from my_runner import MyRunner

CFG = {}

CFG.DESC = 'my net'  # customized description
CFG.RUNNER = MyRunner
CFG.GPU_NUM = 1

CFG.MODEL = {}
CFG.MODEL.NAME = 'my_net'

CFG.TRAIN = {}

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = {}
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.002,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = {}
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.DIR = './my_data'
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True

CFG.VAL = {}

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = {}
CFG.VAL.DATA.DIR = 'mnist_data'

CFG._TRAINING_INDEPENDENT` = [
    'OTHER_CONFIG'
]

```

All configurations consists of two parts:
    1. Training dependent configuration: changing this will affect the training results.
    2. Training independent configuration: changing this will not affect the training results.

Notes:
    All training dependent configurations will be calculated MD5,
    this MD5 value will be the sub directory name of checkpoint save directory.
    If the MD5 value is `098f6bcd4621d373cade4e832627b4f6`,
    real checkpoint save directory is `{CFG.TRAIN.CKPT_SAVE_DIR}/098f6bcd4621d373cade4e832627b4f6`

Notes:
    Each configuration default is training dependent,
    except the key is in `TRAINING_INDEPENDENT_KEYS` or `CFG._TRAINING_INDEPENDENT`
"""
from .config import Config
from .utils import config_str, config_md5, save_config_str, copy_config_file, import_config, convert_config, \
    get_ckpt_save_dir, init_cfg


__all__ = [
    'Config', 'config_str', 'config_md5', 'save_config_str', 'copy_config_file',
    'import_config', 'convert_config', 'get_ckpt_save_dir', 'init_cfg'
]
