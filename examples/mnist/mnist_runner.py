from torch import nn
import torchvision

from easytorch import Runner

from conv_net import ConvNet


class MNISTRunner(Runner):
    def init_training(self, cfg):
        """Initialize training.

        Including loss, meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)

        self.loss = nn.NLLLoss()
        self.loss = self.to_running_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')
        self.register_epoch_meter('val_loss', 'val', '{:.2f}')
        self.register_epoch_meter('val_acc', 'val', '{:.2f}%')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        """Define model.

        If you have multiple models, insert the name and class into the dict below,
        and select it through ```config```.

        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """

        return {
            'conv_net': ConvNet
        }[cfg['MODEL']['NAME']](**cfg['MODEL'].get('PARAM', {}))

    @staticmethod
    def build_train_dataset(cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

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
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

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

    def train_iters(self, epoch, iter_index, data):
        """Training details.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """

        _input, _target = data
        _input = self.to_running_device(_input)
        _target = self.to_running_device(_target)

        output = self.model(_input)
        loss = self.loss(output, _target)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def val_iters(self, iter_index, data):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """

        _input, _target = data
        _input = self.to_running_device(_input)
        _target = self.to_running_device(_target)

        output = self.model(_input)
        pred = output.data.max(1, keepdim=True)[1]
        loss = self.loss(output, _target)
        self.update_epoch_meter('val_loss', loss.item())
        self.update_epoch_meter('val_acc', 100 * pred.eq(_target.data.view_as(pred)).sum())
