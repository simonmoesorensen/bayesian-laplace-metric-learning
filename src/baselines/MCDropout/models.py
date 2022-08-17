import torch
import torch.nn as nn
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet


class SampleNet(nn.Module):
    def pass_through(self, x):
        """ " Normal forward pass"""
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


class MCDropoutHead(SampleNet):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def pass_through(self, x):
        return self.backbone(x)


def MNIST_MCDropout(embedding_size=128):
    """
    Construct a mnist model for MCDropout.
    """
    model = FashionMNISTConvNet(latent_dim=embedding_size)
    model_mc = MCDropoutHead(model)

    return model_mc


def CIFAR10_MCDropout(embedding_size=128):
    """
    Construct a cifar10 model for MCDropout.
    """
    model = CIFAR10ConvNet(latent_dim=embedding_size)
    model_mc = MCDropoutHead(model)

    return model_mc
