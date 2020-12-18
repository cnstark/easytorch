import os
import glob
from abc import ABCMeta, abstractmethod

import torch
from torch import optim
from torch.optim import lr_scheduler

from .utils import *


class EasyTraining:
    __metaclass__ = ABCMeta

    def __init__(self, cfg, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0

        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, cfg.md5())

        self.epoch_meter = {}

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

    @abstractmethod
    def run_iters(self, epoch_index, iter_index, data):
        pass

    def _save_model(self, epoch):
        checkpoint_dict = {}
        checkpoint_dict['epoch'] = epoch
        checkpoint_dict['model_state_dict'] = self.model.state_dict()
        # checkpoint_dict['optim_state_dict'] = self.optim.state_dict()

        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        checkpoint_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        checkpoint_path = os.path.join(self.ckpt_save_dir, checkpoint_name)
        torch.save(checkpoint_dict, checkpoint_path)

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
            ckpt_list.sort()
            ckpt_path = ckpt_list[-1]
        return torch.load(ckpt_path)

    def load_model_resume(self):
        try:
            checkpoint_dict = self._load_checkpoint()
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            # self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
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

    def train(self):
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            print('EPOCH {:d} / {:d}'.format(epoch, self.num_epochs))
            for iter_index, data in enumerate(self.data_loader):
                loss = self.run_iters(epoch, iter_index, data)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            # update lr_scheduler
            if self.scheduler is not None:
                print(self.scheduler.get_lr())
                self.scheduler.step()
            # print meters
            self._print_epoch_meters()
            # save model
            self._save_model(epoch)
            # reset meters
            self._reset_epoch_meters()

    def register_epoch_meter(self, name, fmt='{:f}'):
        self.epoch_meter[name] = {
            'meter': AvgMeter(),
            'index': len(self.epoch_meter.keys()),
            'format': fmt
        }

    def update_epoch_meter(self, name, value):
        self.epoch_meter[name]['meter'].update(value)

    def get_epoch_meter_avg(self, name):
        return self.epoch_meter[name]['meter'].avg

    def _print_epoch_meters(self):
        print_list = []
        for i in range(len(self.epoch_meter.keys())):
            for name, value in self.epoch_meter.items():
                if value['index'] == i:
                    print_list.append(
                        ('{}: ' + value['format']).format(name, value['meter'].avg)
                    )
        print_str = ', '.join(print_list)
        print(print_str)

    def _reset_epoch_meters(self):
        for name, value in self.epoch_meter.items():
            value['meter'].reset()
