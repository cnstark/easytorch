from torch import nn


class ModelTemplate(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)
