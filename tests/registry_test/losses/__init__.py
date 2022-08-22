import os

from torch import nn
from easytorch.utils.registry import scan_modules

from .registry import LOSS_REGISTRY

__all__ = ['LOSS_REGISTRY']

scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])

LOSS_REGISTRY.register(nn.L1Loss, 'L1_LOSS')
LOSS_REGISTRY.register(nn.MSELoss, 'L2_LOSS')
