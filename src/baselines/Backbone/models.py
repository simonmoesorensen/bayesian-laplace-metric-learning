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
        


def MNIST_Backbone(embedding_size=128):
    """
    Construct a mnist model for Backbone.
    """
    # Embedding dimension
    model = resnet18(num_classes=embedding_size)

    # # Adapt to 1 channel inputs
    model.conv1 = Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # model = EmbeddingNet(embedding_size=embedding_size, img_size=28, n_channels=1)
    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model


def CIFAR10_Backbone(embedding_size=128):
    """
    Construct a cifar10 model for Backbone.
    """
    # Embedding dimension
    model = resnet34(pretrained=True)
    # model = EmbeddingNet(embedding_size=embedding_size, img_size=32, n_channels=3)

    embedder = Embedder(model, embedding_size)

    return embedder


def Casia_Backbone(embedding_size=128):
    """
    Construct a Casia Webface model for Backbone.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model
