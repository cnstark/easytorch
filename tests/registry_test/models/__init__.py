import os

from easytorch.utils.registry import scan_modules

from .registry import MODEL_REGISTRY

__all__ = ['MODEL_REGISTRY']

scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])
