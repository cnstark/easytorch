from torch import nn

from easytorch import Runner

from dataset import LinearDataset


class LinearRegressionRunner(Runner):
    def __init__(self, cfg: dict, use_gpu: bool):
        super().__init__(cfg, use_gpu)
        self.loss = None

    def init_training(self, cfg):
        super().init_training(cfg)

        self.loss = nn.MSELoss()
        self.loss = self.to_running_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')
        self.register_epoch_meter('val_loss', 'val', '{:.2f}')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return nn.Linear(1, 1)

    @staticmethod
    def build_train_dataset(cfg: dict):
        return LinearDataset(
            cfg['TRAIN']['DATA']['K'],
            cfg['TRAIN']['DATA']['B'],
            cfg['TRAIN']['DATA']['NUM'],
        )

    @staticmethod
    def build_val_dataset(cfg: dict):
        return LinearDataset(
            cfg['VAL']['DATA']['K'],
            cfg['VAL']['DATA']['B'],
            cfg['VAL']['DATA']['NUM'],
        )

    def train_iters(self, epoch, iter_index, data):
        x, y = data
        x = self.to_running_device(x)
        y = self.to_running_device(y)

        output = self.model(x)
        loss = self.loss(output, y)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def val_iters(self, iter_index, data):
        x, y = data
        x = self.to_running_device(x)
        y = self.to_running_device(y)

        output = self.model(x)
        loss = self.loss(output, y)
        self.update_epoch_meter('val_loss', loss.item())

    def on_training_end(self):
        super().on_training_end()
        self.logger.info('Result: k: {}, b: {}'.format(self.model.weight.item(), self.model.bias.item()))
