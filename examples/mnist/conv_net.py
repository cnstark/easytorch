from torch import nn


class ConvNet(nn.Module):
    """Simple ConvNet for MNIST classification.
    """

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        y = self.conv_block(x)
        y = y.view(-1, 320)
        y = self.fc_block(y)

        return y
