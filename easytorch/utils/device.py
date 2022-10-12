from typing import Union

import torch
from torch import nn

_DEVICE_TYPE = 'gpu'


def set_device_type(device_type: str):
    global _DEVICE_TYPE
    if device_type not in ['gpu', 'mlu', 'cpu']:
        raise ValueError('Unknown device type!')
    _DEVICE_TYPE = device_type


def get_device_type() -> str:
    return _DEVICE_TYPE


def get_device_count() -> int:
    if _DEVICE_TYPE == 'gpu':
        return torch.cuda.device_count()
    elif _DEVICE_TYPE == 'mlu':
        import torch_mlu
        return torch_mlu.mlu_model.device_count()
    elif _DEVICE_TYPE == 'cpu':
        return 0
    else:
        raise ValueError('Unknown device type!')


def set_device(device_id: int):
    if _DEVICE_TYPE == 'gpu':
        torch.cuda.set_device(device_id)
    elif _DEVICE_TYPE == 'mlu':
        import torch_mlu
        torch_mlu.mlu_model.set_device(device_id)
    else:
        raise ValueError('Unknown device type!')


def to_device(src: Union[torch.Tensor, nn.Module], device_id: int = None) -> Union[torch.Tensor, nn.Module]:
    if _DEVICE_TYPE == 'gpu':
        if device_id is None:
            return src.cuda()
        else:
            return src.to('cuda:{:d}'.format(device_id))
    elif _DEVICE_TYPE == 'mlu':
        import torch_mlu
        if device_id is None:
            return src.mlu()
        else:
            return src.to('mlu:{:d}'.format(device_id))
    elif _DEVICE_TYPE == 'cpu':
        return src.cpu()
    else:
        raise ValueError('Unknown device type!')


def set_device_manual_seed(seed: int):
    torch.manual_seed(seed)
    if _DEVICE_TYPE == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif _DEVICE_TYPE == 'mlu':
        import torch_mlu
        torch_mlu.mlu_model.manual_seed(seed)
        torch_mlu.mlu_model.manual_seed_all(seed)
