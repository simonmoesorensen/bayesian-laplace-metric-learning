from torch import nn
from src.layers import L2Normalize


class ConvNet(nn.Module):
    def __init__(self, latent_dim=32, normalize=False):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(0.25),
        #     nn.Flatten(),
        # )
        # linear_layers = [
        #     nn.Linear(6272, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, latent_dim),
        # ]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        ]
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
