import os
import time
import glob
from abc import ABCMeta, abstractmethod

import torch

from .nn_interface import NNInterface
from .config import config_md5

class EasyInfer:
    def __init__(self, cfg, nn: NNInterface):
        self.model_name = cfg.MODEL.NAME
        self.ckpt_save_dir = os.path.join(cfg.TRAIN.CKPT_SAVE_DIR, config_md5(cfg))

        self.nn = nn

        # create model
        self._create_model(cfg)
        # load model
        self.load_model()

        # nn on init
        self.nn.on_init(cfg)

    def _create_model(self, cfg):
        model = self.nn.define_model(cfg)
        model = model.cuda()
        model.eval()
        self.nn.set_model(model)

    def _get_ckpt_path(self):
        ckpt_list = glob.glob(os.path.join(self.ckpt_save_dir, '*.pt'))
        ckpt_list.sort()
        return ckpt_list[-1]

    def _load_checkpoint(self):
        ckpt_path = self._get_ckpt_path()
        return torch.load(ckpt_path)
    
    def load_model(self):
        try:
            checkpoint_dict = self._load_checkpoint()
            self.nn.model.load_state_dict(checkpoint_dict['model_state_dict'])
        except (IndexError, OSError, KeyError):
            raise OSError('ckpt file does not exist')
