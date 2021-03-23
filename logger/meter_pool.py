import torch

from .meter import AvgMeter


METER_TYPES = ['train', 'val']


class MeterPool:
    def __init__(self):
        self._pool = {}

    @property
    def pool(self):
        return self.pool

    @staticmethod
    def _meter_name(meter_type, name):
        return '{}/{}'.format(meter_type, name)

    def register(self, meter_type, name, fmt='{:f}', plt=True):
        if meter_type not in METER_TYPES:
            raise ValueError('Unsupport meter type!')
        self._pool[self._meter_name(meter_type, name)] = {
            'meter': AvgMeter(),
            'index': len(self._pool.keys()),
            'format': fmt,
            'type': meter_type,
            'plt': plt
        }

    def update(self, meter_type, name, value):
        self._pool[self._meter_name(meter_type, name)]['meter'].update(value)

    def get_avg(self, meter_type, name):
        return self._pool[self._meter_name(meter_type, name)]['meter'].avg

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

    def reset_all(self):
        """
        reset all meter
        """
        for _, value in self._pool.items():
            value['meter'].reset()

    def reset(self):
        pass


METER_POOL = MeterPool()
