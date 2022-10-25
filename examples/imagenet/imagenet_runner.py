from typing import Dict, Union, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, datasets, transforms

from easytorch import Runner
from easytorch.device import to_device


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ImagenetRunner(Runner):
    """ImagenetRunner
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = to_device(self.criterion)

    def init_training(self, cfg: Dict):
        super().init_training(cfg)

        self.register_epoch_meter('train/loss', 'train', '{:.4e}')
        self.register_epoch_meter('train/acc@1', 'train', '{:6.2f}')
        self.register_epoch_meter('train/acc@5', 'train', '{:6.2f}')

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)

        self.register_epoch_meter('val/loss', 'val', '{:.4e}')
        self.register_epoch_meter('val/acc@1', 'val', '{:6.2f}')
        self.register_epoch_meter('val/acc@5', 'val', '{:6.2f}')

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return models.__dict__[cfg['MODEL']['NAME']]()

    @staticmethod
    def build_train_dataset(cfg: Dict) -> Dataset:
        normalize = transforms.Normalize(**cfg['TRAIN']['DATA']['NORMALIZE'])
        return datasets.ImageFolder(
            cfg['TRAIN']['DATA']['DIR'],
            transforms.Compose([
                transforms.RandomResizedCrop(cfg['TRAIN']['DATA']['CROP_SIZE']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    @staticmethod
    def build_val_dataset(cfg: Dict):
        normalize = transforms.Normalize(**cfg['VAL']['DATA']['NORMALIZE'])
        return datasets.ImageFolder(
            cfg['VAL']['DATA']['DIR'],
            transforms.Compose([
                transforms.Resize(cfg['VAL']['DATA']['RESIZE']),
                transforms.CenterCrop(cfg['VAL']['DATA']['CROP_SIZE']),
                transforms.ToTensor(),
                normalize,
            ]))

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        images, target = data

        images = to_device(images)
        target = to_device(target)

        output = self.model(images)

        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.update_epoch_meter('train/loss', loss.item(), images.size(0))
        self.update_epoch_meter('train/acc@1', acc1[0], images.size(0))
        self.update_epoch_meter('train/acc@5', acc5[0], images.size(0))

        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        images, target = data

        images = to_device(images)
        target = to_device(target)

        output = self.model(images)

        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.update_epoch_meter('val/loss', loss.item(), images.size(0))
        self.update_epoch_meter('val/acc@1', acc1[0], images.size(0))
        self.update_epoch_meter('val/acc@5', acc5[0], images.size(0))

    def on_validating_end(self, train_epoch: Optional[int]):
        # `None` means validation mode
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/acc@1', greater_best=True)
