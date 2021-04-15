import os
import time
import glob
from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .nn_interface import NNInterface
from .meter_pool import MeterPool
from .config import config_md5, save_config
from.utils.dist import *


class EasyTraining():
    def __init__(self, cfg, nn: NNInterface):
        torch.cuda.set_device(get_rank())

        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0

        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, config_md5(cfg))
        self.over_write_ckpt = cfg.TRAIN.OVERWRITE_CKPT

        self.nn = nn

        self.nn.set_ckpt_save_dir(self.ckpt_save_dir)

        # create model
        self._create_model(cfg)

        # create optim
        self._create_optim(cfg.TRAIN.OPTIM, self.nn.model)

        # create lr_scheduler
        if hasattr(cfg.TRAIN, 'LR_SCHEDULER'):
            self._create_lr_scheduler(cfg.TRAIN.LR_SCHEDULER, self.nn.optim)

        # resume
        if os.path.isdir(self.ckpt_save_dir):
            self._load_model_resume()
        else:
            if is_master():
                os.makedirs(self.ckpt_save_dir)
                save_config(cfg, os.path.join(self.ckpt_save_dir, 'param.txt'))

        # data loader
        if torch.distributed.is_initialized():
            self.train_data_loader = self.nn.define_train_data_loader_ddp(cfg)
        else:
            self.train_data_loader = self.nn.define_train_data_loader(cfg)
        self.val_data_loader = self.nn.define_val_data_loader(cfg)

        # tensorboard & meter_pool
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )
            self._meter_pool = MeterPool(self.tensorboard_writer)
            self.nn.set_tensorboard_writer(self.tensorboard_writer)
            self.nn.set_meter_pool(self._meter_pool)

            # epoch time meter
            self.register_epoch_meter('epoch_time', 'train', '{:.2f} (s)', plt=False)
            self.register_epoch_meter('val_time', 'val', '{:.2f} (s)', plt=False)
            # lr meter
            if self.nn.scheduler is not None:
                self.register_epoch_meter('lr', 'train', '{:.2e}')

        # nn on init
        self.nn.on_init(cfg)

    def _create_model(self, cfg):
        model = self.nn.define_model(cfg)
        model = model.cuda()
        if torch.distributed.is_initialized():
            model = DDP(model, device_ids=[get_rank()])
        self.nn.set_model(model)

    def _create_optim(self, optim_cfg, model):
        Optim = getattr(optim, optim_cfg.TYPE)
        optim_param = optim_cfg.PARAM.copy()
        optimizer = Optim(model.parameters(), **optim_param)
        self.nn.set_optim(optimizer)

    def _create_lr_scheduler(self, lr_scheduler_cfg, optim):
        Scheduler = getattr(lr_scheduler, lr_scheduler_cfg.TYPE)
        scheduler_param = lr_scheduler_cfg.PARAM.copy()
        scheduler_param['optimizer'] = optim
        scheduler = Scheduler(**scheduler_param)
        self.nn.set_scheduler(scheduler)

    def _save_model(self, epoch):
        if isinstance(self.nn.model, DDP):
            _model = self.nn.model.module
        else:
            _model = self.nn.model

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': _model.state_dict(),
            'optim_state_dict': self.nn.optim.state_dict()
        }
        self._save_checkpoint(epoch, checkpoint_dict)

    def _load_model_resume(self):
        try:
            checkpoint_dict = self._load_checkpoint()
            if isinstance(self.nn.model, DDP):
                self.nn.model.module.load_state_dict(checkpoint_dict['model_state_dict'])
            else:
                self.nn.model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.nn.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.nn.scheduler is not None:
                self.nn.scheduler.last_epoch = checkpoint_dict['epoch']
        except (IndexError, OSError, KeyError):
            if is_master():
                print('Resume failed, keeping training')

    def load_model_finetune(self, ckpt_path):
        # TODO
        pass

    def _load_checkpoint(self):
        ckpt_path = self._get_ckpt_path()
        return torch.load(ckpt_path, map_location='cuda:{}'.format(get_rank()))

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

    def _get_ckpt_path(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
        ckpt_list.sort()
        return ckpt_list[-1]

    def train(self):
        if self.train_data_loader == None:
            raise RuntimeError('Please set train data loader')
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self._on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start traning
            self.nn.model.train()
            for iter_index, data in enumerate(self.train_data_loader):
                loss = self.nn.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.nn.backward(loss)
            # update lr_scheduler
            if self.nn.scheduler is not None:
                self.nn.scheduler.step()

            epoch_end_time = time.time()
            self._on_epoch_end(epoch, epoch_end_time - epoch_start_time)
        self._on_training_end()

    @torch.no_grad()
    @master_only
    def validate(self):
        if self.val_data_loader == None:
            raise RuntimeError('Please set val data loader')
        self.nn.model.eval()
        for iter_index, data in enumerate(self.val_data_loader):
            self.nn.val_iters(iter_index, data)

    @master_only
    def _on_epoch_start(self, epoch):
        # print epoch num
        print('EPOCH {:d} / {:d}'.format(epoch, self.num_epochs))
        # update lr meter
        if self.nn.scheduler is not None:
            self.update_epoch_meter('lr', self.nn.scheduler.get_lr()[0])

    @master_only
    def _on_epoch_end(self, epoch, epoch_time):
        # epoch time
        self.update_epoch_meter('epoch_time', epoch_time)
        # print train meters
        self.print_epoch_meters('train')
        # validate
        val_start_time = time.time()
        self.validate()
        val_end_time = time.time()
        self.update_epoch_meter('val_time', val_end_time - val_start_time)
        # print val meters
        self.print_epoch_meters('val')
        # tensorboard plt meters
        self.plt_epoch_meters(epoch)
        # save model
        self._save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    @master_only
    def _on_training_end(self):
        self.nn.on_training_end()

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        self._meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_epoch_meter(self, name, value):
        self._meter_pool.update(name, value)

    @master_only
    def print_epoch_meters(self, meter_type):
        self._meter_pool.print_meters(meter_type)

    @master_only
    def plt_epoch_meters(self, epoch):
        self._meter_pool.plt_meters(epoch)

    @master_only
    def reset_epoch_meters(self):
        self._meter_pool.reset()
