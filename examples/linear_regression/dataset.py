import torch
from torch.utils.data import Dataset


class LinearDataset(Dataset):
    def __init__(self, k: float, b: float, num: int):
        self.num = num
        self.x = torch.linspace(-1, 1, self.num)
        self.y = k * self.x + b + torch.rand(self.x.size())

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.num
