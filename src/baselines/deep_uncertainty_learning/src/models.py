from torchvision.models import resnet18
from torch.nn import Conv2d, BatchNorm1d
import torch.nn as nn

class DUL_Backbone(nn.Module):
    def __init__(self, resnet, embedding_size):
        super(DUL_Backbone, self).__init__()

        self.features = resnet

        self.mu_dul_backbone = nn.Sequential(
            BatchNorm1d(embedding_size),
        )
        self.logvar_dul_backbone = nn.Sequential(
            BatchNorm1d(embedding_size),
        )

    def forward(self, img):
        x = self.features(img)
        mu_dul = self.mu_dul_backbone(x)
        logvar_dul = self.logvar_dul_backbone(x)
        std_dul = (logvar_dul * 0.5).exp()
        return mu_dul, std_dul

def MNIST_DUL(embedding_size=128):
    """
    Construct a mnist model for DUL.
    """
    # Embedding dimension

    model = resnet18(num_classes=embedding_size)
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul

def CIFAR10_DUL(embedding_size=128):
    """
    Construct a cifar10 model for DUL.
    """
    # Embedding dimension

    model = resnet18(num_classes=embedding_size)

    # Wrap in DUL framework
    model_dul = DUL_Backbone(model, embedding_size)

    return model_dul