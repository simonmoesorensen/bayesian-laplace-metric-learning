from turtle import back
import torch
import torch.nn as nn
from src.baselines.models import CIFAR10ConvNet, FashionMNISTLinearNet, FashionMNISTConvNet
from src.utils import filter_state_dict


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

        # Scale and shift parameters from paper
        self.beta = nn.Parameter(torch.zeros(embedding_size) - 7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(embedding_size) * 1e-4, requires_grad=True)

    def scale_and_shift(self, x):
        return self.gamma * x + self.beta

    def forward(self, x):
        # Non-trainable
        x = self.conv(x)

        # Trainable (mu and var)
        mu = self.fc_mu(x)

        # Get log variance
        log_std = self.fc_var(x)
        log_std = self.scale_and_shift(log_std)

        std = log_std.exp()

        return mu, std


def MNIST_PFE(embedding_size=128, seed=42, linear=False):
    """
    Construct a mnist model for PFE.
    """
    if linear:
        # Embedding dimension
        backbone = FashionMNISTLinearNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/MNIST/linaer/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )   
    else:
        # Embedding dimension
        backbone = FashionMNISTConvNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/MNIST/conv/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def FashionMNIST_PFE(embedding_size=128, seed=42, linear=False):
    """
    Construct a fashion mnist model for PFE.
    """
    if linear:
        # Embedding dimension
        backbone = FashionMNISTLinearNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/FashionMNIST/linaer/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )   
    else:
        # Embedding dimension
        backbone = FashionMNISTConvNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/FashionMNIST/conv/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )


    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def CIFAR10_PFE(embedding_size=128, seed=42, linear=False):
    """
    Construct a cifar10 model for PFE.
    """
    if linear:
        # Embedding dimension
        backbone = CIFAR10LinearNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/CIFAR10/linaer/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )   
    else:
        # Embedding dimension
        backbone = CIFAR10ConvNet(latent_dim=embedding_size)
        backbone.load_state_dict(
            filter_state_dict(
                torch.load(
                    "src/baselines/PFE/pretrained/CIFAR10/conv/"
                    f"latentdim_{embedding_size}_seed_{seed}.pth"
                )
            )
        )


    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE


def Casia_PFE(embedding_size=128, seed=42):
    """
    Construct a Casia Webface model for PFE.
    """
    # Embedding dimension
    backbone = Casia_Backbone(embedding_size=embedding_size)
    backbone.load_state_dict(
        filter_state_dict(
            torch.load(
                "src/baselines/PFE/pretrained/CASIA/"
                f"latentdim_{embedding_size}_seed_{seed}.pth"
            )
        )
    )

    # Wrap in PFE framework
    model_PFE = UncertaintyModule(backbone, embedding_size)

    return model_PFE
