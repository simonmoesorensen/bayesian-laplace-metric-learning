import torch.nn as nn
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet
from src.utils import L2Norm
from torchvision. import resnet152


class Embedder(nn.Module):
    def (self, backbone, embedding_size) -> None:
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
    model = FashionMNISTConvNet(latent_dim=embedding_size)

    return model


def CIFAR10_Backbone(embedding_size=128):
    """
    Construct a cifar10 model for Backbone.
    """
    model = CIFAR10ConvNet(latent_dim=embedding_size)

    return model


def Casia_Backbone(embedding_size=128):
    """
    Construct a Casia Webface model for Backbone.
    """
    # Embedding dimension
    model = resnet152(num_classes=embedding_size)

    norm_layer = L2Norm()

    norm_model = nn.Sequential(model, norm_layer)

    return norm_model
