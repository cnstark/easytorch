import functools

import torch


MASTER_RANK = 0


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_rank(rank: int) -> bool:
    if rank >= get_world_size():
        raise ValueError('Rank is out of range')

    return get_rank() == rank


def is_master() -> bool:
    return is_rank(MASTER_RANK)


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)

    return wrapper
