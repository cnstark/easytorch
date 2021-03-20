import types
import hashlib


MD5_EXCEPT_FLAG = '_md5_except'


def except_dict_keys(cfg: dict, except_keys: list):
    cfg_copy = cfg.copy()

    if cfg_copy.get(MD5_EXCEPT_FLAG) is not None:
        cfg_copy.pop(MD5_EXCEPT_FLAG)

    pop_list = []
    dict_list = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            sub_except_keys = []
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)
                elif except_key.find(k) == 0:
                    sub_except_keys.append(except_key[len(k) + 1:])
            if len(sub_except_keys) != 0:
                new_v = except_dict_keys(v, sub_except_keys)
                dict_list.append((k, new_v))
        else:
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)

    for pop_key in pop_list:
        cfg_copy.pop(pop_key)

    for dict_key, dict_value in dict_list:
        cfg_copy[dict_key] = dict_value

    return cfg_copy


def config_str(cfg: dict, indent: str=''):
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


def import_config(path: str, verbose: bool=True):
    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG
    if verbose:
        print_config(cfg)
    return cfg
