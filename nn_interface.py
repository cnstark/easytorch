from abc import ABCMeta, abstractmethod, abstractstaticmethod


class NNInterface(metaclass=ABCMeta):
    def __init__(self):
        self.model = None
        self.optim = None
        self.scheduler = None

        self.meter_pool = None
        self.tensorboard_writer = None

    @abstractstaticmethod
    def define_model(cfg):
        pass

    @abstractstaticmethod
    def define_train_data_loader(cfg):
        pass

    @abstractstaticmethod
    def define_train_data_loader_ddp(cfg, rank, world_size):
        pass

    @abstractstaticmethod
    def define_val_data_loader(cfg):
        pass

    @abstractmethod
    def train_iters(self, epoch, iter_index, data):
        pass

    @abstractmethod
    def val_iters(self, iter_index, data):
        pass

    def on_init(self, cfg):
        pass

    def on_training_end(self):
        pass

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        if self.meter_pool is not None:
            self.meter_pool.register(name, meter_type, fmt, plt)

    def update_epoch_meter(self, name, value):
        if self.meter_pool is not None:
            self.meter_pool.update(name, value)

    def set_model(self, model):
        self.model = model

    def set_optim(self, optimizer):
        self.optim = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_meter_pool(self, meter_pool):
        self.meter_pool = meter_pool
    
    def set_tensorboard_writer(self, tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer
