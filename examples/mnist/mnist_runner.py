from typing import Dict, Union, Tuple

import torch
from torch import nn
import torchvision

from easytorch import Runner

from conv_net import ConvNet


class MNISTRunner(Runner):
    """MNISTRunner
    """

    def init_training(self, cfg: Dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (Dict): config
        """

        super().init_training(cfg)

        self.loss = nn.NLLLoss()
        self.loss = self.to_running_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')

    def init_validation(self, cfg: Dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (Dict): config
        """

        super().init_validation(cfg)

        self.register_epoch_meter('val_acc', 'val', '{:.2f}%')

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        """Define model.

        If you have multiple models, insert the name and class into the dict below,
        and select it through ```config```.

        Args:
            cfg (Dict): config

        Returns:
            model (nn.Module)
        """

        return {
            'conv_net': ConvNet
        }[cfg['MODEL']['NAME']](**cfg['MODEL'].get('PARAM', {}))

    @staticmethod
    def build_train_dataset(cfg: Dict):
        """Build MNIST train dataset

        Args:
            cfg (Dict): config

        Returns:
            train dataset (Dataset)
        """

        return torchvision.datasets.MNIST(
            cfg['TRAIN']['DATA']['DIR'], train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

    @staticmethod
    def build_val_dataset(cfg: Dict):
        """Build MNIST val dataset

        Args:
            cfg (Dict): config

        Returns:
            train dataset (Dataset)
        """

        return torchvision.datasets.MNIST(
            cfg['VAL']['DATA']['DIR'], train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """

        input_, target_ = data
        input_ = self.to_running_device(input_)
        target_ = self.to_running_device(target_)

        output = self.model(input_)
        loss = self.loss(output, target_)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """

        input_, target_ = data
        input_ = self.to_running_device(input_)
        target_ = self.to_running_device(target_)

        output = self.model(input_)
        pred = output.data.max(1, keepdim=True)[1]
        self.update_epoch_meter('val_acc', 100 * pred.eq(target_.data.view_as(pred)).sum())
