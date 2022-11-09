import threading
import queue as Queue

import torch
from torch.utils.data import DataLoader

from .. import device


class BackgroundGenerator(threading.Thread):
    """BackgroundGenerator
    """

    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended
        (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is
        passed through queue.

        There's no restriction on doing weird stuff, reading/writing files,
        retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator
        keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster,
        but will require storing all batches in memory.
        If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)


class DataLoaderX(DataLoader):
    """Dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def data_to_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data_to_device(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = data_to_device(v)
    elif isinstance(data, tuple):
        data = tuple(data_to_device(list(data)))
    elif isinstance(data, torch.Tensor):
        data = device.to_device(data, non_blocking=True)
    return data


class DevicePrefetcher:
    """Device Prefetcher
    """

    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.stream = torch.cuda.Stream()
        self.batch_data = None

    @staticmethod
    def data_to_device(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = device.to_device(v, non_blocking=True)
        elif isinstance(data, (list, tuple)):
            for i, v in enumerate(data):
                if isinstance(v, torch.Tensor):
                    data[i] = device.to_device(v, non_blocking=True)
        elif isinstance(data, torch.Tensor):
            data = device.to_device(data, non_blocking=True)
        return data

    def preload(self):
        try:
            self.batch_data = next(self.data_loader_iter)
            # put tensors to gpu
            with device.stream(self.stream):
                self.batch_data = data_to_device(self.batch_data)
        except StopIteration:
            self.batch_data = None

    def next(self):
        if self.batch_data is None:
            raise StopIteration()

        device.current_stream().wait_stream(self.stream)
        batch = self.batch_data
        self.preload()
        return batch

    def reset(self):
        self.data_loader_iter = iter(self.data_loader)
        self.preload()

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.reset()
        return self

    def __len__(self):
        return len(self.data_loader)
