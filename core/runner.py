import os
import time
from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .meter_pool import MeterPool
from .checkpoint import get_ckpt_dict, load_ckpt, save_ckpt, backup_last_ckpt, clear_ckpt
from .data_loader import build_data_loader, build_data_loader_ddp
from .optimizer_builder import build_optim, build_lr_scheduler
from ..config import config_md5, save_config
from ..utils import get_logger, get_rank, is_master, master_only


class Runner(metaclass=ABCMeta):
    def __init__(self, cfg):
        # param
        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, config_md5(cfg))
        self.ckpt_save_strategy = None
        self.num_epochs = None
        self.start_epoch = None

        self.val_interval = 1

        # default logger
        self.logger = get_logger('easytorch')

        # create model
        self.model = self.build_model(cfg)

        # declare optimizer and lr_scheduler
        self.optim = None
        self.scheduler = None

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None

        # declare meter pool
        self._meter_pool = None

        # declare tensorboard_writer
        self.tensorboard_writer = None

    @staticmethod
    @abstractmethod
    def define_model(cfg):
        pass

    @staticmethod
    @abstractmethod
    def build_train_dataset(cfg: dict):
        pass

    @staticmethod
    @abstractmethod
    def build_val_dataset(cfg: dict):
        pass

    def build_train_data_loader(self, cfg: dict):
        dataset = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg.TRAIN.DATA)
        else:
            return build_data_loader(dataset, cfg.TRAIN.DATA)

    def build_val_data_loader(self, cfg: dict):
        dataset = self.build_val_dataset(cfg)
        return build_data_loader(dataset, cfg.VAL.DATA)

    def build_model(self, cfg):
        model = self.define_model(cfg)
        model = model.cuda()
        if torch.distributed.is_initialized():
            model = DDP(model, device_ids=[get_rank()])
        return model

    def get_ckpt_path(self, epoch: int):
        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    def save_model(self, epoch):
        ckpt_dict = get_ckpt_dict(self.model, self.optim, epoch)

        # backup last epoch
        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # save ckpt
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10 epoch or in the end
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    def _load_model_resume(self, strict=True):
        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('resume training')
        except (IndexError, OSError, KeyError):
            pass

    def load_model(self, ckpt_path=None, strict=True):
        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError):
            raise OSError('Ckpt file does not exist')

    def train(self, cfg):
        self.init_training(cfg)

        # train loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self._on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()
            for iter_index, data in enumerate(self.train_data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            self._on_epoch_end(epoch, epoch_end_time - epoch_start_time)
        self.on_training_end()

    @abstractmethod
    def train_iters(self, epoch, iter_index, data):
        pass

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    @torch.no_grad()
    @master_only
    def validate(self):
        self.model.eval()
        for iter_index, data in enumerate(self.val_data_loader):
            self.val_iters(iter_index, data)

    @abstractmethod
    def val_iters(self, iter_index, data):
        pass

    def init_training(self, cfg):
        # init training param
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0
        if hasattr(cfg.TRAIN, 'CKPT_SAVE_STRATEGY'):
            self.ckpt_save_strategy = cfg.TRAIN.CKPT_SAVE_STRATEGY

        # make ckpt_save_dir
        if is_master() and not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)
            save_config(cfg, os.path.join(self.ckpt_save_dir, 'param.txt'))

        # init logger
        log_file_name = 'training_log_{}.log'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.logger = get_logger('easytorch-training', log_file=os.path.join(self.ckpt_save_dir, log_file_name))

        self.logger.info('ckpt save dir: \'{}\''.format(self.ckpt_save_dir))

        # init meter_pool
        if is_master():
            self._meter_pool = MeterPool()

        # train data loader
        self.train_data_loader = self.build_train_data_loader(cfg)
        self.register_epoch_meter('train_time', 'train', '{:.2f} (s)', plt=False)

        # val config and val data loader
        if hasattr(cfg, 'VAL'):
            if hasattr(cfg.VAL, 'INTERVAL'):
                self.val_interval = cfg.VAL.INTERVAL
            self.val_data_loader = self.build_val_data_loader(cfg)
            self.register_epoch_meter('val_time', 'val', '{:.2f} (s)', plt=False)

        # create optim
        self.optim = build_optim(cfg.TRAIN.OPTIM, self.model)
        self.logger.info('set optim: ' + str(self.optim))

        # create lr_scheduler
        if hasattr(cfg.TRAIN, 'LR_SCHEDULER'):
            self.scheduler = build_lr_scheduler(cfg.TRAIN.LR_SCHEDULER, self.optim)
            self.logger.info('set lr_scheduler: ' + str(self.scheduler))
            self.register_epoch_meter('lr', 'train', '{:.2e}')

        # finetune
        if hasattr(cfg.TRAIN, 'FINETUNE_FROM'):
            self.load_model(cfg.TRAIN.FINETUNE_FROM)
            self.logger.info('start finetuning')

        # resume
        self._load_model_resume()

        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

    @master_only
    def _on_epoch_start(self, epoch):
        # print epoch num
        self.logger.info('epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        # update lr meter
        if self.scheduler is not None:
            self.update_epoch_meter('lr', self.scheduler.get_lr()[0])

    @master_only
    def _on_epoch_end(self, epoch, epoch_time):
        # epoch time
        self.update_epoch_meter('train_time', epoch_time)
        # print train meters
        self.print_epoch_meters('train')
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            val_start_time = time.time()
            self.validate()
            val_end_time = time.time()
            self.update_epoch_meter('val_time', val_end_time - val_start_time)
            # print val meters
            self.print_epoch_meters('val')
        # tensorboard plt meters
        self.plt_epoch_meters(epoch)
        # save model
        self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    def on_training_end(self):
        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        self._meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_epoch_meter(self, name, value):
        self._meter_pool.update(name, value)

    @master_only
    def print_epoch_meters(self, meter_type):
        self._meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_epoch_meters(self, epoch):
        self._meter_pool.plt_meters(epoch, self.tensorboard_writer)

    @master_only
    def reset_epoch_meters(self):
        self._meter_pool.reset()
