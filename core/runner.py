import torch


class Runner:
    def __init__(self):
        pass

    @abstractstaticmethod
    def define_model(cfg):
        pass

    @abstractstaticmethod
    def define_train_dataset(cfg):
        pass

    @abstractstaticmethod
    def define_val_dataset(cfg):
        pass

    @abstractmethod
    def train_iters(self, epoch, iter_index, data):
        pass

    @abstractmethod
    def val_iters(self, iter_index, data):
        pass