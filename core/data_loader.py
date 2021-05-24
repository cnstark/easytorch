from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..utils import get_rank, get_world_size


def build_dataloader(dataset: Dataset, data_cfg: dict):
    if data_cfg.get('PREFETCH', False):
        from ..utils.data_prefetcher import DataLoaderX
        return DataLoaderX(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=data_cfg.get('SHUFFLE', False),
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=data_cfg.get('SHUFFLE', False),
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )


def build_dataloader_ddp(dataset: Dataset, data_cfg: dict):
    ddp_sampler = DistributedSampler(
        dataset,
        get_world_size(),
        get_rank(),
        shuffle=data_cfg.get('SHUFFLE', False)
    )
    if data_cfg.get('PREFETCH', False):
        from ..utils.data_prefetcher import DataLoaderX
        return DataLoaderX(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=False,
            sampler=ddp_sampler,
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=False,
            sampler=ddp_sampler,
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
