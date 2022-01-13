import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, datasets, transforms

from easytorch import Runner
from easytorch.core.checkpoint import get_ckpt_dict, save_ckpt


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
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.to_running_device(self.criterion)

        self.best_acc1 = 0.

    def init_training(self, cfg: dict):
        super().init_training(cfg)

        self.register_epoch_meter('train/loss', 'train', '{:.4e}')
        self.register_epoch_meter('train/acc@1', 'train', '{:6.2f}')
        self.register_epoch_meter('train/acc@5', 'train', '{:6.2f}')

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)

        self.register_epoch_meter('val/loss', 'val', '{:.4e}')
        self.register_epoch_meter('val/acc@1', 'val', '{:6.2f}')
        self.register_epoch_meter('val/acc@5', 'val', '{:6.2f}')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return models.__dict__[cfg['MODEL']['NAME']]()

    @staticmethod
    def build_train_dataset(cfg: dict) -> Dataset:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return datasets.ImageFolder(
            cfg['TRAIN']['DATA']['DIR'],
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    @staticmethod
    def build_val_dataset(cfg: dict):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return datasets.ImageFolder(
            cfg['VAL']['DATA']['DIR'],
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    def train_iters(self, epoch: int, iter_index: int, data: torch.Tensor or tuple) -> torch.Tensor:
        images, target = data

        images = self.to_running_device(images)
        target = self.to_running_device(target)

        output = self.model(images)

        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.update_epoch_meter('train/loss', loss.item(), images.size(0))
        self.update_epoch_meter('train/acc@1', acc1[0], images.size(0))
        self.update_epoch_meter('train/acc@5', acc5[0], images.size(0))

        return loss

    def val_iters(self, iter_index: int, data: torch.Tensor or tuple):
        images, target = data

        images = self.to_running_device(images)
        target = self.to_running_device(target)

        output = self.model(images)

        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.update_epoch_meter('val/loss', loss.item(), images.size(0))
        self.update_epoch_meter('val/acc@1', acc1[0], images.size(0))
        self.update_epoch_meter('val/acc@5', acc5[0], images.size(0))

    def save_best_model(self, epoch: int, acc1: float):
        if acc1 > self.best_acc1:
            self.best_acc1 = acc1

            ckpt_dict = get_ckpt_dict(self.model, self.optim, epoch)
            ckpt_path = os.path.join(self.ckpt_save_dir, '{}_best.pt'.format(self.model_name))
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
