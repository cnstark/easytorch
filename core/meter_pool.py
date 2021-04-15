import torch


METER_TYPES = ['train', 'val']


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MeterPool:
    def __init__(self):
        self._pool = {}

    def register(self, name, meter_type, fmt='{:f}', plt=True):
        if meter_type not in METER_TYPES:
            raise ValueError('Unsupport meter type!')
        self._pool[name] = {
            'meter': AvgMeter(),
            'index': len(self._pool.keys()),
            'format': fmt,
            'type': meter_type,
            'plt': plt
        }

    def update(self, name, value):
        self._pool[name]['meter'].update(value)

    def get_avg(self, name):
        return self._pool[name]['meter'].avg

    def print_meters(self, meter_type):
        print_list = []
        for i in range(len(self._pool.keys())):
            for name, value in self._pool.items():
                if value['index'] == i and value['type'] == meter_type:
                    print_list.append(
                        ('{}: ' + value['format']).format(name, value['meter'].avg)
                    )
        print_str = '{}:: [{}]'.format(meter_type, ', '.join(print_list))
        print(print_str)

    def plt_meters(self, epoch, tensorboard_writer):
        for name, value in self._pool.items():
            if value['plt']:
                tensorboard_writer.add_scalar(name, value['meter'].avg, global_step=epoch)

    def reset(self):
        for _, value in self._pool.items():
            value['meter'].reset()


class MeterPoolDDP(MeterPool):
    # TODO(Yuhao Wang): not support

    def to_tensor(self):
        tensor = torch.empty((len(self._pool.keys()), 2))
        for i in range(len(self._pool.keys())):
            for _, value in self._pool.items():
                if value['index'] == i:
                    tensor[i][0] = float(value['meter'].count)
                    tensor[i][1] = value['meter'].avg
        return tensor
    
    def update_tensor(self, tensor):
        if tensor.shape[0] != len(self._pool.keys()):
            raise ValueError('Invalid tensor shape!')
        for i in range(len(self._pool.keys())):
            for _, value in self._pool.items():
                if value['index'] == i:
                    value['meter'].update(tensor[i][1], tensor[i][0])
