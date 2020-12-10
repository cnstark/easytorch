import types
from easydict import EasyDict


class Config(EasyDict):
    def _print_cfg(self, indent=''):
        for k, v in self.items():
            if isinstance(v, Config):
                print((indent + '{}:').format(k))
                v._print_cfg(indent + '  ')
            elif isinstance(v, (types.MethodType, types.FunctionType)):
                pass
            else:
                print((indent + '{}: {}').format(k, v))

    def pure_dict(self):
        d = self.copy()

        pop_list = []
        for k, v in d.items():
            if isinstance(v, (types.MethodType, types.FunctionType)):
                pop_list.append(k)

        for pop_key in pop_list:
            d.pop(pop_key)
        
        return d

    @staticmethod
    def import_cfg(path, verbose=True):
        cfg_name = path.split('.')[-1]
        cfg = __import__(path, fromlist=[cfg_name]).CFG
        if verbose:
            cfg._print_cfg()
        return cfg
