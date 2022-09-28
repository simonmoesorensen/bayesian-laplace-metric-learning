import torch
import torch.nn as nn
from src.baselines.models import CIFAR10ConvNet, CIFAR10LinearNet, FashionMNISTConvNet, FashionMNISTLinearNet, CUB200ConvNet
from torch.nn import BatchNorm1d
from torchvision.models import resnet152

class UncertaintyModule(nn.Module):
    """
    Uncertainty Module
    """

    def __init__(self, backbone, embedding_size=128):
        super(UncertaintyModule, self).__init__()

        # Conv part
        self.conv = backbone.conv
        
        # Linear part
        self.fc_mu = backbone.linear

        self.fc_var = nn.Sequential(
            nn.Linear(backbone.linear[0].in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        
        torch.nn.init.constant_(self.fc_var[3].weight, 1e-6)
        torch.nn.init.constant_(self.fc_var[3].bias, 1e-6)

    def forward(self, x):
        # Non-trainable
        x = self.conv(x)

        # Trainable (mu and var)
        mu = self.fc_mu(x)

        # Get log variance
        log_std = self.fc_var(x)

        std = log_std.exp()

        return mu, std

def MNIST_DUL(embedding_size=128, linear=False):
    """
    Construct a mnist model for DUL.
    """
    # Embedding dimension
    if linear:
        backbone = FashionMNISTLinearNet(latent_dim=embedding_size)
        
        # we need to move some of the layers from backbone.linear to backbone.conv
        linear_new = backbone.linear[-4:]
        conv_new = nn.Sequential(*[backbone.conv[0]] + list(backbone.linear[:-4]))
        
        backbone.linear = linear_new
        backbone.conv = conv_new
    else:
        model = FashionMNISTConvNet(latent_dim=embedding_size)

    # Wrap in DUL framework
    model = UncertaintyModule(model, embedding_size)

    return model


def CIFAR10_DUL(embedding_size=128, linear=False):
    """
    Construct a cifar10 model for DUL.
    """
    # Embedding dimension
    if linear:
        model = CIFAR10LinearNet(latent_dim=embedding_size)
    else:
        model = CIFAR10ConvNet(latent_dim=embedding_size)

    # Wrap in DUL framework
    model = UncertaintyModule(model, embedding_size)

    return model


def Casia_DUL(embedding_size=128, linear=False):
    """
    Construct a Casia Webface model for DUL.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    # Wrap in DUL framework
    model = UncertaintyModule(model, embedding_size)

    return model

def Cub200_DUL(embedding_size=128, linear=False):
    
    if linear:
        raise NotImplementedError
    
    model = CUB200ConvNet(embedding_size)
    
    model = UncertaintyModule(model, embedding_size)
    
    return model
