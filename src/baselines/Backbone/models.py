from torchvision.models import resnet18, resnet34, resnet152
from torch.nn import Conv2d
import torch.nn as nn
from src.baselines.models import EmbeddingNet

from src.utils import L2Norm


class Embedder(nn.Module):
    def __init__(self, backbone, embedding_size) -> None:
        super().__init__()

        no_last_layer = list(backbone.children())[:-1]
        last_layer_size = backbone.fc.in_features

        norm_layer = L2Norm()

        self.backbone = nn.Sequential(*no_last_layer)

        self.norm_model = nn.Sequential(
            nn.Linear(last_layer_size, embedding_size),
            norm_layer,
        )

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)

        return self.norm_model(out)


class ConvNet(nn.Module):
    def __init__(self, n_channels=3, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(6272, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim),
        ]

        # norm_layer = L2Norm()

        # self.linear = nn.Sequential(*linear_layers, norm_layer)
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class FashionMNISTConvNet(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        linear_layers = [
            nn.Linear(4608, latent_dim),
        ]
        # norm_layer = L2Norm()

        # self.linear = nn.Sequential(*linear_layers, norm_layer)
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


def MNIST_Backbone(embedding_size=128):
    """
    Construct a mnist model for Backbone.
    """
    # Embedding dimension
    # model = resnet18(num_classes=embedding_size)

    # # Adapt to 1 channel inputs
    # model.conv1 = Conv2d(
    # 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    # )

    model = FashionMNISTConvNet(latent_dim=embedding_size)
    # norm_layer = L2Norm()

    # norm_model = nn.Sequential(model, norm_layer)

    return model


def CIFAR10_Backbone(embedding_size=128):
    """
    Construct a cifar10 model for Backbone.
    """
    # Embedding dimension
    # model = resnet34(pretrained=True)
    # # model = EmbeddingNet(embedding_size=embedding_size, img_size=32, n_channels=3)

    # embedder = Embedder(model, embedding_size)

    # return embedder

    model = ConvNet(latent_dim=embedding_size)

    return model


def Casia_Backbone(embedding_size=128):
    """
    Construct a Casia Webface model for Backbone.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    # norm_layer = L2Norm()

    # norm_model = nn.Sequential(model, norm_layer)
    norm_model = nn.Sequential(model)

    return norm_model
