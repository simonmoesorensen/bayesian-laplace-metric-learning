from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import Conv2d, BatchNorm1d
import torch.nn as nn
import torch

from src.utils import l2_norm


class UncertaintyModule(nn.Module):
    """
    Uncertainty Module
    """

    def __init__(self, resnet, embedding_size=128):
        super(UncertaintyModule, self).__init__()

        # Freeze resnet parameters
        for param in resnet.parameters():
            param.requires_grad = False

        resnet_no_last_layer = list(resnet.children())[:-1]
        last_layer_size = resnet.fc.in_features

        self.bottleneck = nn.Sequential(*resnet_no_last_layer)

        self.fc_mu = nn.Linear(last_layer_size, embedding_size)

        self.fc_var1 = nn.Sequential(
            nn.Linear(last_layer_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.fc_var2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.beta = nn.Parameter(torch.zeros(embedding_size))
        self.gamma = nn.Parameter(torch.ones(embedding_size))

        self.relu = nn.ReLU()

    def scale_and_shift(self, x):
        return self.gamma * x + self.beta

    def forward(self, x):
        # Non-trainable
        mu = self.bottleneck(x)

        # Trainable (mu and var)
        mu = self.fc_mu(mu.view(mu.size(0), -1))
        mu = l2_norm(mu)

        # Get log var
        log_var = self.bottleneck(x)
        log_var = self.relu(self.fc_var1(log_var.view(log_var.size(0), -1)))
        log_var = self.fc_var2(log_var)

        log_var = self.scale_and_shift(log_var)

        std = (log_var * 0.5).exp()

        return mu, std


def MNIST_PFE(embedding_size=128):
    """
    Construct a mnist model for PFE.
    """
    # Embedding dimension
    model = resnet50(pretrained=True)

    # Adapt to 1 channel inputs
    model.conv1 = Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(model, embedding_size)

    return model_PFE


def CIFAR10_PFE(embedding_size=128):
    """
    Construct a cifar10 model for PFE.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(model, embedding_size)

    return model_PFE


def Casia_PFE(embedding_size=128):
    """
    Construct a Casia Webface model for PFE.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(model, embedding_size)

    return model_PFE
