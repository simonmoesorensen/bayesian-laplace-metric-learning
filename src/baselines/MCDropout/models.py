from torch.nn import Conv2d
import torch.nn as nn
import torch

from src.utils import L2Norm

class SampleNet(nn.Module):
    def pass_through(self, x):
        """" Normal forward pass"""
        raise NotImplementedError()

    def sample(self, X, samples):
        zs = []

        for _ in range(samples):
            zs.append(self.pass_through(X))
        
        zs = torch.stack(zs)

        return zs.mean(dim=0), zs.std(dim=0)

    def forward(self, x, samples=100):
        if samples:
            return self.sample(x, samples)
        else:
            return self.pass_through(x), None


class Cifar10ConvNet(SampleNet):
    def __init__(self, n_channels=3, latent_dim=128, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, 1),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(6272, 256),
            nn.Dropout(p=p),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Dropout(p=p),
            nn.Tanh(),
            nn.Linear(256, latent_dim),
        ]

        norm_layer = L2Norm()

        self.linear = nn.Sequential(*linear_layers, norm_layer)

    def pass_through(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class FashionMNISTConvNet(SampleNet):
    def __init__(self, latent_dim=32, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.Dropout(p=p),
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


    def pass_through(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x



def MNIST_MCDropout(embedding_size=128):
    """
    Construct a mnist model for MCDropout.
    """
    model = FashionMNISTConvNet(latent_dim=embedding_size)
    return model


def CIFAR10_MCDropout(embedding_size=128):
    """
    Construct a cifar10 model for MCDropout.
    """
    model = Cifar10ConvNet(latent_dim=embedding_size)

    return model
