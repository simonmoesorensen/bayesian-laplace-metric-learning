from torchvision.models import resnet50
from torch.nn import Conv2d, BatchNorm1d
import torch.nn as nn

from src.utils import l2_norm


class DUL_Backbone(nn.Module):
    def __init__(self, resnet, embedding_size):
        super(DUL_Backbone, self).__init__()

        self.features = resnet

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            BatchNorm1d(embedding_size),
            nn.Dropout(0.4)
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            BatchNorm1d(embedding_size),
            nn.Dropout(0.4)
        )

    def forward(self, img):
        x = self.features(img)
        mu_dul = self.mu_dul_backbone(x)
        logvar_dul = self.logvar_dul_backbone(x)
        std_dul = (logvar_dul * 0.5).exp()

        mu_dul = l2_norm(mu_dul)
        return mu_dul, std_dul


def MNIST_DUL(embedding_size=128):
    """
    Construct a mnist model for DUL.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Adapt to 1 channel inputs
    model.conv1 = Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul


def CIFAR10_DUL(embedding_size=128):
    """
    Construct a cifar10 model for DUL.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul


def Casia_DUL(embedding_size=128):
    """
    Construct a Casia Webface model for DUL.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul
