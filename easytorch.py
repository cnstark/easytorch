import os
import time
import glob
from abc import ABCMeta, abstractmethod

import torch
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from .meter_pool import MeterPool
from .utils import AvgMeter


class EasyTraining:
    __metaclass__ = ABCMeta

    def __init__(self, cfg, model, train_data_loader, val_data_loader):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0

        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, cfg.md5())
        self.over_write_ckpt = cfg.TRAIN.OVERWRITE_CKPT

        Optim = getattr(optim, cfg.TRAIN.OPTIM.TYPE)
        optim_param = cfg.TRAIN.OPTIM.PARAM.pure_dict()
        self.optim = Optim(self.model.parameters(), **optim_param)

        self.model = self.model.cuda()

        if hasattr(cfg.TRAIN, 'LR_SCHEDULER'):
            Scheduler = getattr(lr_scheduler, cfg.TRAIN.LR_SCHEDULER.TYPE)
            scheduler_param = cfg.TRAIN.LR_SCHEDULER.PARAM.pure_dict()
            scheduler_param['optimizer'] = self.optim
            self.scheduler = Scheduler(**scheduler_param)
        else:
            self.scheduler = None

        if os.path.isdir(self.ckpt_save_dir):
            self.load_model_resume()
        else:
            os.makedirs(self.ckpt_save_dir)
            cfg.export(os.path.join(self.ckpt_save_dir, 'param.txt'))

        tensorboard_dir = os.path.join(self.ckpt_save_dir, 'tensorboard')
        self.tensorboard_writer = SummaryWriter(
            tensorboard_dir,
            purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
        )

        self._meter_pool = MeterPool(self.tensorboard_writer)

        self.register_epoch_meter('epoch_time', 'train', '{:.2f} (s)', plt=False)

    def _save_model(self, epoch):
        if self.over_write_ckpt:
            self._backup_checkpoint()
        checkpoint_dict = {}
        checkpoint_dict['epoch'] = epoch
        checkpoint_dict['model_state_dict'] = self.model.state_dict()
        checkpoint_dict['optim_state_dict'] = self.optim.state_dict()

        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        checkpoint_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        checkpoint_path = os.path.join(self.ckpt_save_dir, checkpoint_name)
        torch.save(checkpoint_dict, checkpoint_path)
        if self.over_write_ckpt:
            self._clear_checkpoint()

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
            ckpt_list.sort()
            ckpt_path = ckpt_list[-1]
        return torch.load(ckpt_path)

    def _clear_checkpoint(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt.bak'))
        for ckpt in ckpt_list:
            os.remove(ckpt)

    def _backup_checkpoint(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
        for ckpt in ckpt_list:
            ckpt_bak = ckpt + '.bak'
            os.rename(ckpt, ckpt_bak)

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

    def _iter_report(self, epoch_index, iter_index):
        # TODO
        pass

    @abstractmethod
    def run_iters(self, epoch_index, iter_index, data):
        pass

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            print('EPOCH {:d} / {:d}'.format(epoch, self.num_epochs))
            epoch_start_time = time.time()
            for iter_index, data in enumerate(self.train_data_loader):
                loss = self.run_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                print(self.scheduler.get_lr())
                self.scheduler.step()

            # epoch time
            epoch_end_time = time.time()
            self.update_epoch_meter('epoch_time', epoch_end_time - epoch_start_time)
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

    @abstractmethod
    def val_iters(self, iter_index, data):
        pass

    def validate(self):
        for iter_index, data in enumerate(self.val_data_loader):
            self.val_iters(iter_index, data)

    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        self._meter_pool.register(name, meter_type, fmt, plt)
    
    def update_epoch_meter(self, name, value):
        self._meter_pool.update(name, value)
