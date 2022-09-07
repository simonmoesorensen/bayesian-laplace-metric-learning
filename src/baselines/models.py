import torch.nn as nn
import torch


class SampleNet(nn.Module):
    def pass_through(self, x):
        """ " Normal forward pass"""
        raise NotImplementedError()

    def sample(self, X, samples):
        zs = []

        for _ in range(samples):
            zs.append(self.pass_through(X))

        zs = torch.stack(zs, dim=-1)

        return zs.mean(dim=-1), zs.std(dim=-1), zs

    def forward(self, x, samples=100):
        if samples:
            return self.sample(x, samples)
        else:
            return self.pass_through(x)


class CIFAR10ConvNet(nn.Module):
    def __init__(self, latent_dim=128, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, latent_dim),
        ]

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class FashionMNISTConvNet(nn.Module):
    def __init__(self, latent_dim=32, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(3 * 3 * 32, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        ]

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
