from torch import nn

from src.layers import L2Normalize


class MNIST(nn.Module):
    pass


class CIFAR10(nn.Module):
    def __init__(self, embedding_size=32, normalize=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(6272, 64),
            nn.Tanh(),
            nn.Linear(64, embedding_size),
        ]
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class Casia(nn.Module):
    pass
