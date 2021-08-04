import torch
from torch.utils.data import Dataset


class LinearDataset(Dataset):
    def __init__(self, k: float, b: float, num: int):
        self.num = num
        self.x = torch.unsqueeze(torch.linspace(-1, 1, self.num), dim=1)
        self.y = k * self.x + b + torch.rand(self.x.size()) - 0.5

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num
