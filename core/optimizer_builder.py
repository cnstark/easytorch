from torch import nn, optim
from torch.optim import lr_scheduler

from ..easyoptim import easy_lr_scheduler


def build_optim(optim_cfg: dict, model: nn.Module):
    Optim = getattr(optim, optim_cfg.TYPE)
    optim_param = optim_cfg.PARAM.copy()
    optimizer = Optim(model.parameters(), **optim_param)
    return optimizer


def build_lr_scheduler(lr_scheduler_cfg: dict, optimizer: optim.Optimizer):
    if hasattr(lr_scheduler, lr_scheduler_cfg.TYPE):
        Scheduler = getattr(lr_scheduler, lr_scheduler_cfg.TYPE)
    else:
        Scheduler = getattr(easy_lr_scheduler, lr_scheduler_cfg.TYPE)
    scheduler_param = lr_scheduler_cfg.PARAM.copy()
    scheduler_param['optimizer'] = optimizer
    scheduler = Scheduler(**scheduler_param)
    return scheduler
