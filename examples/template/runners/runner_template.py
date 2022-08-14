from typing import Dict

import torch
from torch import nn
from torch.utils.data import Dataset

from easytorch import Runner

from ..models import MODEL_DICT


class RunnerTemplate(Runner):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_training(self, cfg: Dict):
        super().init_training(cfg)

        # init loss
        # e.g.
        # self.loss = nn.MSELoss()
        # self.loss = self.to_running_device(self.loss)

        # register meters by calling:
        # self.register_epoch_meter('train_loss', 'train', '{:.2f}')

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)

        # self.register_epoch_meter('val_acc', 'val', '{:.2f}%')

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return MODEL_DICT[cfg['MODEL']['NAME']](**cfg['MODEL'].get('PARAM', {}))

    @staticmethod
    def build_train_dataset(cfg: Dict) -> Dataset:
        # return your train Dataset
        pass

    @staticmethod
    def build_val_dataset(cfg: Dict):
        # return your val Dataset
        pass

    def train_iters(self, epoch: int, iter_index: int, data: torch.Tensor or tuple) -> torch.Tensor:
        # forward and compute loss
        # update meters if necessary
        # return loss (will be auto backward and update params) or don't return any thing

        # e.g.
        # _input, _target = data
        # _input = self.to_running_device(_input)
        # _target = self.to_running_device(_target)
        #
        # output = self.model(_input)
        # loss = self.loss(output, _target)
        # self.update_epoch_meter('train_loss', loss.item())
        # return loss
        pass

    def val_iters(self, iter_index: int, data: torch.Tensor or tuple):
        # forward and update meters
        pass
