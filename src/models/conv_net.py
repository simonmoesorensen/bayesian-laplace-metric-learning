from torch import nn
from src.laplace.layers import Norm2, Reciprocal, Sqrt


class ConvNet(nn.Module):
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
        self.linear = nn.Sequential(
            nn.Linear(6272, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.normalizer = nn.Sequential(
            Norm2(dim=1),
            Sqrt(),
            Reciprocal(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        x = x * self.normalizer(x)
        return x
