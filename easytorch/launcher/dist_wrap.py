import functools
import random
from typing import Callable, Dict, Union, Any, Optional

import torch

from ..utils import get_logger
from ..device import get_device_type, set_device_type, get_device_count, set_device


def dist_func(local_rank: int, dist_params: Dict[str, Any], func: Callable, *args):
    """Distributed function for `torch.multiprocessing.spawn`

    Args:
        local_rank (int): Local rank of current process group.
        dist_params (Dict[str, Any]): Other distributed parameters.
        func (Callable): A function.
    """

    logger = get_logger('easytorch-launcher')

    rank = dist_params['device_num'] * dist_params['node_rank'] + local_rank
    logger.info(
        'Launching in distributed mode. Distributed parameters:'\
        'word_size={:d}, node_rank={:d}, rank={:d}, local_rank={:d}, dist_backend={}, init_method={}'.format(
            dist_params['word_size'], dist_params['node_rank'], rank, local_rank,
            dist_params['dist_backend'], dist_params['init_method']
        )
    )

    set_device_type(dist_params['device_type'])

    torch.distributed.init_process_group(
        backend=dist_params['dist_backend'],
        init_method=dist_params['init_method'],
        rank=rank,
        world_size=dist_params['word_size']
    )

    set_device(local_rank)

    args, kwargs = args
    func(*args, **kwargs)


def dist_wrap(func: Callable,
        node_num: int = 1,
        device_num: int = 1,
        node_rank: int = 0,
        dist_backend: Optional[Union[str, torch.distributed.Backend]] = None,
        init_method: Optional[str] = None) -> Callable:
    """Convert a function to a distributed function.

    Usage:
        >>> def function(a, b):
        >>>     ...
        >>>
        >>> function_dist = dist_wrap(
        >>>     function,
        >>>     node_num=node_num,
        >>>     device_num=device_num,
        >>>     node_rank=node_rank,
        >>>     dist_backend=dist_backend,
        >>>     init_method=init_method
        >>> )
        >>> function_dist(a, b)

    Args:
        func (Callable): The function.
        node_num (int, optional): Number of node. Defaults to 1.
        device_num (int, optional): Number of devices per node. Defaults to 1.
        node_rank (int, optional): Rank of current node. Defaults to 0.
        dist_backend (Optional[Union[str, distributed.Backend]], optional): The backend of DDP.
            Defaults to None, means using `nccl` as the backend.
        init_method (Optional[str], optional): URL specifying how to initialize the process group.
            Defaults to None, means using `172.0.0.1:{random port}` as the init method.

    Returns:
        Callable: The converted function.
    """

    if node_num < 1:
        raise ValueError('The node_num must be greater than 1!')

    if device_num < 0:
        raise ValueError('The device_num must be greater than 0!')

    word_size = node_num * device_num

    if word_size == 0:
        # CPU mode
        return func
    else:
        # DEVICE mode
        if node_rank >= node_num:
            raise ValueError('The node_rank must be less than dist_node_num!')

        if device_num != get_device_count():
            raise RuntimeError('Device num not match, cfg.DEVICE_NUM = {:d}, ' \
                'but torch.cuda.device_count() = {:d}'.format(device_num, get_device_count()))

        if word_size == 1:
            return func
        else:
            # Distributed Data Parallel
            dist_backend = 'nccl' if dist_backend is None else dist_backend

            if init_method is None:
                if node_num == 1:
                    init_method = 'tcp://127.0.0.1:{:d}'.format(random.randint(50000, 65000))
                else:
                    raise ValueError('The init_method cannot be None in multiple compute nodes')

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                dist_params = {
                    'device_type': get_device_type(),
                    'device_num': device_num,
                    'node_rank': node_rank,
                    'word_size': word_size,
                    'dist_backend': dist_backend,
                    'init_method': init_method
                }

                torch.multiprocessing.spawn(
                    dist_func,
                    args=(dist_params, func, args, kwargs),
                    nprocs=device_num,
                    join=True
                )

            return wrapper
