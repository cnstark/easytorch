"""Everything is based on config.

`Config` is the set of all configurations. `Config` is is implemented by `dict`, We recommend using `EasyDict`.

Look at the following example:

cfg.py

```python
import os
from easydict import EasyDict

from my_runner import MyRunner

CFG = EasyDict()

CFG.DESC = 'my net'  # customized description
CFG.RUNNER = MyRunner
CFG.GPU_NUM = 1

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'my_net'

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 100
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
CFG.TRAIN.DATA.DIR = './my_data'
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True

CFG.VAL = EasyDict()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = 'mnist_data'

```

All configurations consists of two parts:
the configurations that affects the training results and the configurations that does not affect the training results

"""

import types
import hashlib

MD5_EXCEPT_FLAG = '_md5_except'

DEFAULT_MD5_EXCEPT_KEYS = [
    'TRAIN.CKPT_SAVE_STRATEGY',
    'TRAIN.DATA.NUM_WORKERS',
    'TRAIN.DATA.PIN_MEMORY',
    'TRAIN.DATA.PREFETCH',
    'VAL'
]


def except_dict_keys(cfg: dict, except_keys: set):
    """

    Args:
        cfg:
        except_keys:

    Returns:

    """
    cfg_copy = cfg.copy()

    if cfg_copy.get(MD5_EXCEPT_FLAG) is not None:
        cfg_copy.pop(MD5_EXCEPT_FLAG)

    pop_list = []
    dict_list = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            sub_except_keys = set([])
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)
                elif except_key.find(k) == 0 and except_key[len(k)] == '.':
                    sub_except_keys.add(except_key[len(k) + 1:])
            if len(sub_except_keys) != 0:
                new_v = except_dict_keys(v, sub_except_keys)
                dict_list.append((k, new_v))
        else:
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)

    for dict_key, dict_value in dict_list:
        cfg_copy[dict_key] = dict_value

    for pop_key in pop_list:
        cfg_copy.pop(pop_key)

    return cfg_copy


def config_str(cfg: dict, indent: str = ''):
    s = ''
    for k, v in cfg.items():
        if isinstance(v, dict):
            s += (indent + '{}:').format(k) + '\n'
            s += config_str(v, indent + '  ')
        elif isinstance(v, types.FunctionType):
            s += (indent + '{}: {}').format(k, v.__name__) + '\n'
        elif k == MD5_EXCEPT_FLAG:
            pass
        else:
            s += (indent + '{}: {}').format(k, v) + '\n'
    return s


def config_md5(cfg: dict):
    cfg_excepted = except_dict_keys(cfg, cfg[MD5_EXCEPT_FLAG])
    m = hashlib.md5()
    m.update(config_str(cfg_excepted).encode('utf-8'))
    return m.hexdigest()


def print_config(cfg: dict):
    print('MD5: {}'.format(config_md5(cfg)))
    print(config_str(cfg))


def save_config(cfg: dict, file_path: str):
    with open(file_path, 'w') as f:
        content = 'MD5: {}\n'.format(config_md5(cfg))
        content += config_str(cfg)
        f.write(content)


def import_config(path: str, verbose: bool = True):
    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG

    # merge default md5 except keys
    if cfg.get(MD5_EXCEPT_FLAG) is None:
        cfg[MD5_EXCEPT_FLAG] = set([])
    else:
        cfg[MD5_EXCEPT_FLAG] = set(cfg[MD5_EXCEPT_FLAG])
    for k in DEFAULT_MD5_EXCEPT_KEYS:
        cfg[MD5_EXCEPT_FLAG].add(k)

    if verbose:
        print_config(cfg)
    return cfg
