import torch.nn as nn
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet
from src.utils import L2Norm
from torchvision.models import resnet152


def MNIST_Backbone(embedding_size=128):
    """
    Construct a mnist model for Backbone.
    """
    model = FashionMNISTConvNet(latent_dim=embedding_size)

    l2norm = L2Norm()
    model_norm = nn.Sequential(model, l2norm)

    return model_norm


def CIFAR10_Backbone(embedding_size=128):
    """
    Construct a cifar10 model for Backbone.
    """
    model = CIFAR10ConvNet(latent_dim=embedding_size)

    l2norm = L2Norm()
    model_norm = nn.Sequential(model, l2norm)

    return model_norm


def Casia_Backbone(embedding_size=128):
    """
    Construct a Casia Webface model for Backbone.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    l2norm = L2Norm()
    model_norm = nn.Sequential(model, l2norm)

    return model_norm
