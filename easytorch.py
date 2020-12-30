import os
import time
import glob
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from .meter_pool import MeterPool


class _BaseEasyTraining(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.train_data_loader = self.define_train_data_loader(cfg)
        self.val_data_loader = self.define_val_data_loader(cfg)

        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0

        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, cfg.md5())
        self.over_write_ckpt = cfg.TRAIN.OVERWRITE_CKPT

        # create model
        self.model = self._create_model(cfg)

        # create optim
        self.optim = self._create_optim(cfg.TRAIN.OPTIM, self.model)

        # create lr_scheduler
        if hasattr(cfg.TRAIN, 'LR_SCHEDULER'):
            self.scheduler = self._create_lr_scheduler(cfg.TRAIN.LR_SCHEDULER, self.optim)
        else:
            self.scheduler = None

    def _create_model(self, cfg):
        model = self.define_model(cfg)
        model = model.cuda()
        return self.model_decorator(model)

    def model_decorator(self, model):
        return model

    @staticmethod
    @abstractmethod
    def define_model(cfg):
        pass

    @abstractmethod
    def define_train_data_loader(self, cfg):
        pass

    @abstractmethod
    def define_val_data_loader(self, cfg):
        pass

    @staticmethod
    def _create_optim(optim_cfg, model):
        Optim = getattr(optim, optim_cfg.TYPE)
        optim_param = optim_cfg.PARAM.pure_dict()
        return Optim(model.parameters(), **optim_param)

    @staticmethod
    def _create_lr_scheduler(lr_scheduler_cfg, optim):
        Scheduler = getattr(lr_scheduler, lr_scheduler_cfg.TYPE)
        scheduler_param = lr_scheduler_cfg.PARAM.pure_dict()
        scheduler_param['optimizer'] = optim
        return Scheduler(**scheduler_param)

    def _save_checkpoint(self, epoch, checkpoint_dict):
        if self.over_write_ckpt:
            self._backup_checkpoint()
        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        checkpoint_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        checkpoint_path = os.path.join(self.ckpt_save_dir, checkpoint_name)
        torch.save(checkpoint_dict, checkpoint_path)
        if self.over_write_ckpt:
            self._clear_checkpoint()

    def _clear_checkpoint(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt.bak'))
        for ckpt in ckpt_list:
            os.remove(ckpt)

    def _backup_checkpoint(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
        for ckpt in ckpt_list:
            ckpt_bak = ckpt + '.bak'
            os.rename(ckpt, ckpt_bak)

    def _iter_report(self, epoch_index, iter_index):
        # TODO
        pass

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self._on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start traning
            self.model.train()
            for iter_index, data in enumerate(self.train_data_loader):
                loss = self.run_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            self._on_epoch_end(epoch, epoch_end_time - epoch_start_time)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for iter_index, data in enumerate(self.val_data_loader):
                self.val_iters(iter_index, data)

    @abstractmethod
    def _on_epoch_start(self, epoch):
        pass

    @abstractmethod
    def _on_epoch_end(self, epoch, epoch_time):
        pass

    @abstractmethod
    def run_iters(self, epoch, iter_index, data):
        pass

    @abstractmethod
    def val_iters(self, iter_index, data):
        pass


class EasyTraining(_BaseEasyTraining, metaclass=ABCMeta):

    def __init__(self, cfg):
        super(EasyTraining, self).__init__(cfg)
        # resume
        if os.path.isdir(self.ckpt_save_dir):
            self.load_model_resume()
        else:
            os.makedirs(self.ckpt_save_dir)
            cfg.export(os.path.join(self.ckpt_save_dir, 'param.txt'))

        # tensorboard & meter_pool
        self.tensorboard_writer = SummaryWriter(
            os.path.join(self.ckpt_save_dir, 'tensorboard'),
            purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
        )
        self._meter_pool = MeterPool(self.tensorboard_writer)
        # epoch time meter
        self.register_epoch_meter('epoch_time', 'train', '{:.2f} (s)', plt=False)
        # lr meter
        if self.scheduler is not None:
            self.register_epoch_meter('lr', 'train', '{:.2e}')

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
            ckpt_list.sort()
            ckpt_path = ckpt_list[-1]
        return torch.load(ckpt_path)

    def _save_model(self, epoch):
        checkpoint_dict = {}
        checkpoint_dict['epoch'] = epoch
        checkpoint_dict['model_state_dict'] = self.model.state_dict()
        checkpoint_dict['optim_state_dict'] = self.optim.state_dict()
        self._save_checkpoint(epoch, checkpoint_dict)

    def load_model_resume(self):
        try:
            checkpoint_dict = self._load_checkpoint()
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
        except (IndexError, OSError, KeyError):
            print('Resume failed, keeping training')

    def load_model_finetune(self, ckpt_path):
        checkpoint_dict = self._load_checkpoint(ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])

    def _on_epoch_start(self, epoch):
        print('EPOCH {:d} / {:d}'.format(epoch, self.num_epochs))
        if self.scheduler is not None:
            self.update_epoch_meter('lr', self.scheduler.get_lr()[0])

    def _on_epoch_end(self, epoch, epoch_time):
        # epoch time
        self.update_epoch_meter('epoch_time', epoch_time)
        # print train meters
        self._meter_pool.print_meters('train')
        # validate
        self.validate()
        # print val meters
        self._meter_pool.print_meters('val')
        # tensorboard plt meters
        self._meter_pool.plt_meters(epoch)
        # save model
        self._save_model(epoch)
        # reset meters
        self._meter_pool.reset()

    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        self._meter_pool.register(name, meter_type, fmt, plt)

    def update_epoch_meter(self, name, value):
        self._meter_pool.update(name, value)


class EasyTrainingDDP(_BaseEasyTraining, metaclass=ABCMeta):

    def __init__(self, cfg, rank, world_size):
        torch.cuda.set_device(rank)
        self.rank = rank
        self.world_size = world_size
        super(EasyTrainingDDP, self).__init__(cfg)
        # resume
        if os.path.isdir(self.ckpt_save_dir):
            self.load_model_resume()
        else:
            if self.rank == 0:
                os.makedirs(self.ckpt_save_dir)
                cfg.export(os.path.join(self.ckpt_save_dir, 'param.txt'))

        # tensorboard & meter_pool
        if self.rank == 0:
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )
            self._meter_pool = MeterPool(self.tensorboard_writer)
            # epoch time meter
            self.register_epoch_meter('epoch_time', 'train', '{:.2f} (s)', plt=False)
            # lr meter
            if self.scheduler is not None:
                self.register_epoch_meter('lr', 'train', '{:.2e}')

    def model_decorator(self, model):
        return nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])

    def _save_model(self, epoch):
        checkpoint_dict = {}
        checkpoint_dict['epoch'] = epoch
        checkpoint_dict['model_state_dict'] = self.model.module.state_dict()
        checkpoint_dict['optim_state_dict'] = self.optim.state_dict()
        self._save_checkpoint(epoch, checkpoint_dict)

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
            ckpt_list.sort()
            ckpt_path = ckpt_list[-1]
        return torch.load(ckpt_path, map_location='cuda:{}'.format(self.rank))

    def load_model_resume(self):
        try:
            checkpoint_dict = self._load_checkpoint()
            self.model.module.load_state_dict(checkpoint_dict['model_state_dict'])
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
        except (IndexError, OSError, KeyError):
            if self.rank == 0:
                print('Resume failed, keeping training')

    def load_model_finetune(self, ckpt_path):
        checkpoint_dict = self._load_checkpoint(ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])

    def _on_epoch_start(self, epoch):
        if self.rank == 0:
            print('EPOCH {:d} / {:d}'.format(epoch, self.num_epochs))
            if self.scheduler is not None:
                self.update_epoch_meter('lr', self.scheduler.get_lr()[0])

    def _on_epoch_end(self, epoch, epoch_time):
        if self.rank == 0:
            # epoch time
            self.update_epoch_meter('epoch_time', epoch_time)
            # print train meters
            self._meter_pool.print_meters('train')
            # validate
            self.validate()
            # print val meters
            self._meter_pool.print_meters('val')
            # tensorboard plt meters
            self._meter_pool.plt_meters(epoch)
            # save model
            self._save_model(epoch)
            # reset meters
            self._meter_pool.reset()

    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        if self.rank == 0:
            self._meter_pool.register(name, meter_type, fmt, plt)

    def update_epoch_meter(self, name, value):
        if self.rank == 0:
            self._meter_pool.update(name, value)
