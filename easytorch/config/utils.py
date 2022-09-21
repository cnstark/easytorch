import os
import shutil
import types
import copy
import hashlib
from typing import Dict, Set, List, Union

from .config import Config

__all__ = [
    'config_str', 'config_md5', 'save_config_str', 'copy_config_file',
    'import_config', 'convert_config', 'get_ckpt_save_dir'
]

TRAINING_INDEPENDENT_FLAG = '_TRAINING_INDEPENDENT'

TRAINING_INDEPENDENT_KEYS = {
    'DIST_BACKEND',
    'DIST_INIT_METHOD',
    'TRAIN.CKPT_SAVE_STRATEGY',
    'TRAIN.DATA.NUM_WORKERS',
    'TRAIN.DATA.PIN_MEMORY',
    'TRAIN.DATA.PREFETCH',
    'VAL'
}


def get_training_dependent_config(cfg: Dict, except_keys: Union[Set, List] = None) -> Dict:
    """Get training dependent config.
    Recursively traversal each key,
    if the key is in `TRAINING_INDEPENDENT_KEYS` or `CFG._TRAINING_INDEPENDENT`, pop it.

    Args:
        cfg (Dict): Config
        except_keys (Union[Set, List]): the keys need to be excepted

    Returns:
        cfg (Dict): Training dependent configs
    """
    cfg_copy = copy.deepcopy(cfg)

    if except_keys is None:
        except_keys = copy.deepcopy(TRAINING_INDEPENDENT_KEYS)
        if cfg_copy.get(TRAINING_INDEPENDENT_FLAG) is not None:
            except_keys.update(cfg_copy[TRAINING_INDEPENDENT_FLAG])

    # convert to set
    if isinstance(except_keys, list):
        except_keys = set(except_keys)

    if cfg_copy.get(TRAINING_INDEPENDENT_FLAG) is not None:
        cfg_copy.pop(TRAINING_INDEPENDENT_FLAG)

    pop_list = []
    dict_list = []
    for k, v in cfg_copy.items():
        if isinstance(v, dict):
            sub_except_keys = set([])
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)
                elif except_key.find(k) == 0 and except_key[len(k)] == '.':
                    sub_except_keys.add(except_key[len(k) + 1:])
            if len(sub_except_keys) != 0:
                new_v = get_training_dependent_config(v, sub_except_keys)
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


def config_str(cfg: Dict, indent: str = '') -> str:
    """Get config string

    Args:
        cfg (Dict): Config
        indent (str): if ``cfg`` is a sub config, ``indent`` += '    '

    Returns:
        Config string (str)
    """

    s = ''
    for k, v in cfg.items():
        if isinstance(v, dict):
            s += (indent + '{}:').format(k) + '\n'
            s += config_str(v, indent + '  ')
        elif isinstance(v, types.FunctionType):
            s += (indent + '{}: {}').format(k, v.__name__) + '\n'
        elif k == TRAINING_INDEPENDENT_FLAG:
            pass
        else:
            s += (indent + '{}: {}').format(k, v) + '\n'
    return s


def config_md5(cfg: Dict) -> str:
    """Get MD5 value of config.

    Notes:
        Only training dependent configurations participate in the MD5 calculation.

    Args:
        cfg (Dict): Config

    Returns:
        MD5 (str)
    """

    cfg_excepted = get_training_dependent_config(cfg)
    m = hashlib.md5()
    m.update(config_str(cfg_excepted).encode('utf-8'))
    return m.hexdigest()


def save_config_str(cfg: Dict, file_path: str):
    """Save config

    Args:
        cfg (Dict): Config
        file_path (str): file path
    """

    with open(file_path, 'w') as f:
        f.write(config_str(cfg))


def copy_config_file(cfg_file_path: str, save_dir: str):
    """Copy config file to `save_dir`

    Args:
        cfg_file_path (str): config file path
        save_dir (str): save directory
    """

    if os.path.isfile(cfg_file_path) and os.path.isdir(save_dir):
        cfg_file_name = os.path.basename(cfg_file_path)
        shutil.copyfile(cfg_file_path, os.path.join(save_dir, cfg_file_name))


def import_config(path: str, verbose: bool = True) -> Dict:
    """Import config by path

    Examples:
        ```
        cfg = import_config('config/my_config.py')
        ```
        is equivalent to
        ```
        from config.my_config import CFG as cfg
        ```

    Args:
        path (str): Config path
        verbose (str): set to ``True`` to print config

    Returns:
        cfg (Dict): `CFG` in config file
    """

    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG

    if verbose:
        print(config_str(cfg))
    return cfg


def convert_config(cfg: Dict) -> Config:
    """Convert cfg to `Config`; add MD5 to cfg.

    Args:
        cfg (Dict): config.
    """

    if not isinstance(cfg, Config):
        cfg = Config(cfg)
    if cfg.get('MD5') is None:
        cfg['MD5'] = config_md5(cfg)
    return cfg


def get_ckpt_save_dir(cfg: Dict) -> str:
    """Get real ckpt save dir with MD5.

    Args:
        cfg (Dict): config.

    Returns:
        str: Real ckpt save dir
    """

    return os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], cfg['MD5'])


def init_cfg(cfg: Union[Dict, str], save: bool = False):
    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg, verbose=save)
    else:
        cfg_path = None

    # convert ckpt save dir
    cfg = convert_config(cfg)

    # save config
    ckpt_save_dir = get_ckpt_save_dir(cfg)
    if save and not os.path.isdir(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
        save_config_str(cfg, os.path.join(ckpt_save_dir, 'cfg.txt'))
        if cfg_path is not None:
            copy_config_file(cfg_path, ckpt_save_dir)

    return cfg
