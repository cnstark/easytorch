import os
import time
import glob
from abc import ABCMeta, abstractmethod, abstractstaticmethod

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .meter_pool import MeterPool
from ..config import config_md5, save_config
from ..utils import get_logger, get_rank, is_master, master_only
from ..easyoptim import easy_lr_scheduler


class Runner(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.start_epoch = 0

        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, config_md5(cfg))
        self.ckpt_save_strategy = cfg.TRAIN.CKPT_SAVE_STRATEGY if hasattr(cfg.TRAIN, 'CKPT_SAVE_STRATEGY') else None

        # create model
        self.model = self._create_model(cfg)

        # data loader
        if torch.distributed.is_initialized():
            self.train_data_loader = self.define_train_data_loader_ddp(cfg)
        else:
            self.train_data_loader = self.define_train_data_loader(cfg)

        # val config
        if hasattr(cfg, 'VAL'):
            self.val_interval = cfg.VAL.INTERVAL if hasattr(cfg.VAL, 'INTERVAL') else 1
            self.val_data_loader = self.define_val_data_loader(cfg)
        else:
            self.val_data_loader = None

        # init meter_pool
        if is_master():
            self._meter_pool = MeterPool()

        # init time meter
        self.register_epoch_meter('train_time', 'train', '{:.2f} (s)', plt=False)
        self.register_epoch_meter('val_time', 'val', '{:.2f} (s)', plt=False)

    @abstractstaticmethod
    def define_model(cfg):
        pass

    @abstractstaticmethod
    def define_train_data_loader(cfg):
        pass

    @abstractstaticmethod
    def define_train_data_loader_ddp(cfg):
        pass

    @abstractstaticmethod
    def define_val_data_loader(cfg):
        pass

    def _create_model(self, cfg):
        model = self.define_model(cfg)
        model = model.cuda()
        if torch.distributed.is_initialized():
            model = DDP(model, device_ids=[get_rank()])
        return model

    @staticmethod
    def _create_optim(optim_cfg, model):
        Optim = getattr(optim, optim_cfg.TYPE)
        optim_param = optim_cfg.PARAM.copy()
        optimizer = Optim(model.parameters(), **optim_param)
        return optimizer

    @staticmethod
    def _create_lr_scheduler(lr_scheduler_cfg, optim):
        if hasattr(lr_scheduler, lr_scheduler_cfg.TYPE):
            Scheduler = getattr(lr_scheduler, lr_scheduler_cfg.TYPE)
        else:
            Scheduler = getattr(easy_lr_scheduler, lr_scheduler_cfg.TYPE)
        scheduler_param = lr_scheduler_cfg.PARAM.copy()
        scheduler_param['optimizer'] = optim
        scheduler = Scheduler(**scheduler_param)
        return scheduler

    def _save_model(self, epoch):
        if isinstance(self.model, DDP):
            _model = self.model.module
        else:
            _model = self.model

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': _model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }
        self._save_checkpoint(epoch, checkpoint_dict)

    def _load_model_resume(self):
        try:
            ckpt_path = self._get_ckpt_path()
            self.logger.info('load ckpt from \'{}\''.format(ckpt_path))
            checkpoint_dict = torch.load(ckpt_path, map_location='cuda:{}'.format(get_rank()))
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('resume training')
        except (IndexError, OSError, KeyError):
            pass

    def _load_model_finetune(self, ckpt_path):
        self.logger.info('load ckpt from \'{}\''.format(ckpt_path))
        checkpoint_dict = torch.load(ckpt_path, map_location='cuda:{}'.format(get_rank()))
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.logger.info('start finetuning')

    def load_model_inference(self, ckpt_path=None):
        try:
            if ckpt_path is None:
                ckpt_path = self._get_ckpt_path()
            self.logger.info('load ckpt from \'{}\''.format(ckpt_path))
            checkpoint_dict = torch.load(ckpt_path, map_location='cuda:{}'.format(get_rank()))
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        except (IndexError, OSError, KeyError):
            raise OSError('ckpt file does not exist')

    def _save_checkpoint(self, epoch, checkpoint_dict):
        last_epoch = epoch - 1

        # ckpt save strategy
        if self.ckpt_save_strategy is None:
            remove_last_epoch = True
        elif isinstance(self.ckpt_save_strategy, int) and last_epoch % self.ckpt_save_strategy != 0:
            remove_last_epoch = True
        elif isinstance(self.ckpt_save_strategy, list) and not last_epoch in self.ckpt_save_strategy:
            remove_last_epoch = True
        else:
            remove_last_epoch = False

        # rename last ckpt to .bak
        if remove_last_epoch and last_epoch != 0:
            last_epoch_str = str(last_epoch).zfill(len(str(self.num_epochs)))
            last_ckpt_name = '{}_{}.pt'.format(self.model_name, last_epoch_str)
            last_ckpt_path = os.path.join(self.ckpt_save_dir, last_ckpt_name)
            os.rename(last_ckpt_path, last_ckpt_path + '.bak')

        # save ckpt
        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        checkpoint_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        checkpoint_path = os.path.join(self.ckpt_save_dir, checkpoint_name)
        torch.save(checkpoint_dict, checkpoint_path)
        self.logger.info('ckpt {} saved'.format(checkpoint_path))

        # clear ckpt every 10 epoch or in the end
        if epoch % 10 == 0 or epoch == self.num_epochs:
            ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt.bak'))
            for ckpt in ckpt_list:
                os.remove(ckpt)

    def _get_ckpt_path(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
        ckpt_list.sort()
        return ckpt_list[-1]

    def train(self, cfg):
        # make ckpt_save_dir
        if is_master() and not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)
            save_config(cfg, os.path.join(self.ckpt_save_dir, 'param.txt'))

        log_file_name = 'training_log_{}.log'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.logger = get_logger('easytorch-training', log_file=os.path.join(self.ckpt_save_dir, log_file_name))

        # self.logger.info('training config:\n{}'.format(config_str(cfg)))
        self.logger.info('ckpt save dir: \'{}\''.format(self.ckpt_save_dir))

        # create optim
        self.optim = self._create_optim(cfg.TRAIN.OPTIM, self.model)
        self.logger.info('set optim: ' + str(self.optim))

        # create lr_scheduler
        if hasattr(cfg.TRAIN, 'LR_SCHEDULER'):
            self.scheduler = self._create_lr_scheduler(cfg.TRAIN.LR_SCHEDULER, self.optim)
            self.logger.info('set lr_scheduler: ' + str(self.scheduler))
            self.register_epoch_meter('lr', 'train', '{:.2e}')

        # finetune
        if hasattr(cfg.TRAIN, 'FINETUNE_FROM'):
            self._load_model_finetune(cfg.TRAIN.FINETUNE_FROM)

        # resume
        self._load_model_resume()

        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

        # train loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self._on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start traning
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
        self._save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    def on_training_end(self):
        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()
            self.logger.info('tensorboard_writer closed')

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
