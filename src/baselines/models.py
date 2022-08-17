import torch.nn as nn
from src.utils import L2Norm


class CIFAR10ConvNet(nn.Module):
    def __init__(self, latent_dim=128):
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
            nn.Linear(6272, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim),
        ]

        norm_layer = L2Norm()

        self.linear = nn.Sequential(*linear_layers, norm_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class FashionMNISTConvNet(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(4608, latent_dim),
        ]
        norm_layer = L2Norm()

        self.linear = nn.Sequential(*linear_layers, norm_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
