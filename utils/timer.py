import time
from collections import OrderedDict


class Timer:
    def __init__(self):
        self._record_dict = {'Start': time.time()}
        self._record_names = ['Start']

    def record(self, name: str=None):
        if name is None:
            name = 'Record_{:d}'.format(len(self._record_names))
        elif self._record_dict.get(name) is not None:
            raise ValueError('Name \'{}\' already exists'.format(name))

        self._record_dict[name] = time.time()
        self._record_names.append(name)

    def print(self):
        start_time_record = last_time_record = self._record_dict['Start']
        for name in self._record_names:
            time_record = self._record_dict[name]
            time_diff = time_record - last_time_record
            time_total = time_record - start_time_record
            last_time_record = time_record
            print('{}:: [diff: {:2f}, total: {:2f}]'.format(name, time_diff, time_total))

    def get(self, end: str or int, start: str or int=None):
        # end
        if isinstance(end, int):
            end_record_index = end
            end_record_name = self._record_names[end_record_index]
        else:
            end_record_name = end
            end_record_index = self._record_names.index(end_record_name)
        end_record_time = self._record_dict[end_record_name]

        # start
        if start is None:
            start_record_index = max(end_record_index - 1, 0)
            start_record_name = self._record_names[start_record_index]
        elif isinstance(start, int):
            start_record_name = self._record_names[start]
        else:
            start_record_name = start
        start_record_time = self._record_dict[start_record_name]

        return end_record_time - start_record_time, end_record_time - self._record_dict['Start']
