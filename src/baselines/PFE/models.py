import torch
import torch.nn as nn
from src.baselines.Backbone.models import (
    Casia_Backbone,
    CIFAR10_Backbone,
    MNIST_Backbone,
)


class UncertaintyModule(nn.Module):
    """
    Uncertainty Module
    """

    def __init__(self, backbone, embedding_size=128):
        super(UncertaintyModule, self).__init__()

        # Freeze backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False

        # Conv part
        backbone_no_last_layer = list(backbone.children())[:-1][0].conv
        # Linear part without last layer
        backbone_no_last_layer.append(list(backbone.children())[:-1][0].linear[:-1])

        last_layer_size = backbone[0].linear[-1].in_features

        # Define bottleneck as model without the last layer
        self.bottleneck = nn.Sequential(*backbone_no_last_layer)

        # Use pretrained last layer (fully connected) to compute mu with l2norm
        self.fc_mu = nn.Sequential(backbone[0].linear[-1], backbone[1])

        self.fc_var1 = nn.Sequential(
            nn.Linear(last_layer_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.fc_var2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        # Scale and shift parameters from paper
        self.beta = nn.Parameter(torch.zeros(embedding_size) - 7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(embedding_size) * 1e-4, requires_grad=True)

        self.relu = nn.ReLU()

    def scale_and_shift(self, x):
        return self.gamma * x + self.beta

    def forward(self, x):
        # Non-trainable
        mu = self.bottleneck(x)

        # Trainable (mu and var)
        mu = self.fc_mu(mu.view(mu.size(0), -1))

        # Get log var
        log_var = self.bottleneck(x)
        log_var = self.relu(self.fc_var1(log_var.view(log_var.size(0), -1)))
        log_var = self.fc_var2(log_var)

        log_var = self.scale_and_shift(log_var)

        # Numerical stableness
        log_var = (1e-6 + log_var.exp()).log()

        std = (log_var * 0.5).exp()

        return mu, std


def MNIST_PFE(embedding_size=128):
    """
    Construct a mnist model for PFE.
    """
    # Embedding dimension
    backbone = MNIST_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(torch.load("src/baselines/PFE/pretrained/mnist.pth"))

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def FashionMNIST_PFE(embedding_size=128):
    """
    Construct a fashion mnist model for PFE.
    """
    # Embedding dimension
    backbone = MNIST_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(
        torch.load(
            f"src/baselines/PFE/pretrained/fashion_mnist_latent_{embedding_size}.pth"
        )
    )

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def CIFAR10_PFE(embedding_size=128):
    """
    Construct a cifar10 model for PFE.
    """
    # Embedding dimension
    backbone = CIFAR10_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(
        torch.load(f"src/baselines/PFE/pretrained/cifar10_latent_{embedding_size}.pth")
    )

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def Casia_PFE(embedding_size=128):
    """
    Construct a Casia Webface model for PFE.
    """
    # Embedding dimension
    backbone = Casia_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(torch.load("src/baselines/PFE/pretrained/casia.pth"))

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE
