from torchvision.models import resnet50
from torch.nn import Conv2d
import torch.nn as nn

from src.utils import L2Norm


def MNIST_Backbone(embedding_size=128):
    """
    Construct a mnist model for Backbone.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Adapt to 1 channel inputs
    model.conv1 = Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model


def CIFAR10_Backbone(embedding_size=128):
    """
    Construct a cifar10 model for Backbone.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model


def Casia_Backbone(embedding_size=128):
    """
    Construct a Casia Webface model for Backbone.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model
