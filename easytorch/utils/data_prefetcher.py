from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """Dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
