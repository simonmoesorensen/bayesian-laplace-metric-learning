import torch.nn as nn
import torch


class SampleNet(nn.Module):
    def pass_through(self, x):
        """ " Normal forward pass"""
        raise NotImplementedError()

    def sample(self, X, samples):
        zs = self.get_samples(X, samples)

        return zs.mean(dim=0), zs.std(dim=0)

    def get_samples(self, X, samples):
        zs = []

        for _ in range(samples):
            zs.append(self.pass_through(X))

        zs = torch.stack(zs)
        return zs

    def forward(self, x, samples=100):
        if samples:
            return self.sample(x, samples)
        else:
            return self.pass_through(x), None


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

        self.linear = nn.Sequential(*linear_layers)

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

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
