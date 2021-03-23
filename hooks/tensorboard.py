from .hook import Hook
from ..logger.meter_pool import METER_POOL
from ..core.dist import master_only


class TensorboardHook(Hook):
    def __init__(self):
        self._tensorboard_writer = None

    def plt_meters(self, epoch):
        for name, value in METER_POOL.pool.items():
            if value['plt']:
                self._tensorboard_writer.add_scalar(name, value['meter'].avg, global_step=epoch)

    @master_only
    def after_train_epoch(self, runner):
        pass

    @master_only
    def after_train_iter(self, runner):
        pass

    @master_only
    def after_val(self, runner):
        pass
