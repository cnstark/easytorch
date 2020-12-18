import types
import hashlib
from easydict import EasyDict


class Config(EasyDict):
    def _cfg_str(self, indent=''):
        s = ''
        for k, v in self.items():
            if isinstance(v, Config):
                s += (indent + '{}:').format(k) + '\n'
                s += v._cfg_str(indent + '  ')
            elif isinstance(v, (types.MethodType, types.FunctionType)):
                pass
            else:
                s += (indent + '{}: {}').format(k, v) + '\n'
        return s

    def print_cfg(self, indent=''):
        print('MD5: {}'.format(self.md5()))
        print(self._cfg_str())

    def export(self, file_path):
        with open(file_path, 'w') as f:
            content = 'MD5: {}\n'.format(self.md5())
            content += self._cfg_str()
            f.write(content)

    def pure_dict(self):
        d = self.copy()

        pop_list = []
        dict_list = []
        for k, v in d.items():
            if isinstance(v, (types.MethodType, types.FunctionType)):
                pop_list.append(k)
            elif isinstance(v, Config):
                dict_list.append(k)

        for pop_key in pop_list:
            d.pop(pop_key)

        for dict_key in dict_list:
            d[dict_key] = d[dict_key].pure_dict()

        return d

    @staticmethod
    def import_cfg(path, verbose=True):
        cfg_name = path.split('.')[-1]
        cfg = __import__(path, fromlist=[cfg_name]).CFG
        if verbose:
            cfg.print_cfg()
        return cfg

    def md5(self):
        m = hashlib.md5()
        m.update(str(sorted(self.pure_dict())).encode('utf-8'))
        return m.hexdigest()
