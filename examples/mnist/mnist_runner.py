from torch import nn
import torchvision

from easytorch import Runner

from conv_net import ConvNet


class MNISTRunner(Runner):
    def __init__(self, cfg: dict, use_gpu: bool):
        super().__init__(cfg, use_gpu)
        self.loss = None

    def init_training(self, cfg):
        super().init_training(cfg)

        self.loss = nn.NLLLoss()
        self.loss = self.to_running_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')
        self.register_epoch_meter('val_loss', 'val', '{:.2f}')
        self.register_epoch_meter('val_acc', 'val', '{:.2f}%')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return {
            'conv_net': ConvNet
        }[cfg['MODEL']['NAME']](**cfg['MODEL'].get('PARAM', {}))

    @staticmethod
    def build_train_dataset(cfg: dict):
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
        return torchvision.datasets.MNIST(
            cfg['VAL']['DATA']['DIR'], train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

    def train_iters(self, epoch, iter_index, data):
        _input, _target = data
        _input = self.to_running_device(_input)
        _target = self.to_running_device(_target)

        output = self.model(_input)
        loss = self.loss(output, _target)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def val_iters(self, iter_index, data):
        _input, _target = data
        _input = self.to_running_device(_input)
        _target = self.to_running_device(_target)

        output = self.model(_input)
        pred = output.data.max(1, keepdim=True)[1]
        loss = self.loss(output, _target)
        self.update_epoch_meter('val_loss', loss.item())
        self.update_epoch_meter('val_acc', 100 * pred.eq(_target.data.view_as(pred)).sum())
