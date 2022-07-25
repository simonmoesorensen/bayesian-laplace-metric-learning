from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import Conv2d, BatchNorm1d
import torch.nn as nn
import torch
from src.baselines.Backbone.models import CIFAR10_Backbone, Casia_Backbone, MNIST_Backbone


class UncertaintyModule(nn.Module):
    """
    Uncertainty Module
    """

    def __init__(self, backbone, embedding_size=128):
        super(UncertaintyModule, self).__init__()

        # Freeze backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False

        resnet = backbone[0]
        l2norm = backbone[1]

        backbone_no_last_layer = list(resnet.children())[:-1]
        last_layer_size = resnet.fc.in_features

        # Define bottleneck as model without the last layer
        self.bottleneck = nn.Sequential(*backbone_no_last_layer)

        # Use pretrained last layer (fully connected, l2norm) to compute mu
        self.fc_mu = resnet.fc
        self.l2_norm = l2norm

        self.fc_var1 = nn.Sequential(
            nn.Linear(last_layer_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.fc_var2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        # Scale and shift parameters from original code
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
        mu = self.l2_norm(mu)

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


def CIFAR10_PFE(embedding_size=128):
    """
    Construct a cifar10 model for PFE.
    """
    # Embedding dimension
    backbone = CIFAR10_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(torch.load("src/baselines/PFE/pretrained/cifar10.pth"))

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
