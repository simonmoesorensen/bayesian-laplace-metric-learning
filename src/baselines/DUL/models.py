import torch
import torch.nn as nn
from src.baselines import CIFAR10ConvNet, FashionMNISTConvNet
from src.utils import l2_norm
from torch.nn import BatchNorm1d
from torchvision.models import resnet152


class DUL_Backbone(nn.Module):
    def __init__(self, resnet, embedding_size):
        super(DUL_Backbone, self).__init__()

        self.features = resnet

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            BatchNorm1d(embedding_size),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            BatchNorm1d(embedding_size),
        )

        torch.nn.init.constant_(self.logvar_dul_backbone[0].weight, 1e-6)

    def forward(self, img):
        x = self.features(img)
        mu_dul = self.mu_dul_backbone(x)
        logvar_dul = self.logvar_dul_backbone(x)

        # Numerical stability
        logvar_dul = (1e-6 + logvar_dul.exp()).log()

        std_dul = (logvar_dul * 0.5).exp()

        mu_dul = l2_norm(mu_dul)

        return mu_dul, std_dul


def MNIST_DUL(embedding_size=128):
    """
    Construct a mnist model for DUL.
    """
    # Embedding dimension
    model = FashionMNISTConvNet(latent_dim=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul


def CIFAR10_DUL(embedding_size=128):
    """
    Construct a cifar10 model for DUL.
    """
    # Embedding dimension
    model = CIFAR10ConvNet(latent_dim=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul


def Casia_DUL(embedding_size=128):
    """
    Construct a Casia Webface model for DUL.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul
