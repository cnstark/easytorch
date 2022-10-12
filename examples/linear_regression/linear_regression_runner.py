from torch import nn

from easytorch import Runner, to_device

from dataset import LinearDataset


class LinearRegressionRunner(Runner):
    """LinearRegressionRunner
    """

    def init_training(self, cfg):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)

        self.loss = nn.MSELoss()
        self.loss = to_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        """Define model.

        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """

        return nn.Linear(1, 1)

    @staticmethod
    def build_train_dataset(cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        return LinearDataset(
            cfg['TRAIN']['DATA']['K'],
            cfg['TRAIN']['DATA']['B'],
            cfg['TRAIN']['DATA']['NUM'],
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

        x, y = data
        x = to_device(x)
        y = to_device(y)

        output = self.model(x)
        loss = self.loss(output, y)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def on_training_end(self):
        """Print result on training end.
        """

        super().on_training_end()
        self.logger.info('Result: k: {}, b: {}'.format(self.model.weight.item(), self.model.bias.item()))
