from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import Conv2d, BatchNorm1d
import torch.nn as nn




class StochasticLayer(nn.Module):
    """
    Stochastic layer.
    """
    def __init__(self, embedding_size):
        super(StochasticLayer, self).__init__()
        
        self.fc_mu = nn.Linear(embedding_size, embedding_size)
        self.fc_std = nn.Linear(embedding_size, embedding_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        mu = self.fc_mu(x)
        std = self.relu(self.fc_std(x))

        return mu, std

def MNIST_HIB(embedding_size=128):
    """
    Construct a mnist model for HIB.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    # Adapt to 1 channel inputs
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model_HIB = nn.Sequential(
        model,
        StochasticLayer(
            embedding_size=embedding_size
        )
    )

    return model_HIB

def CIFAR10_HIB(embedding_size=128):
    """
    Construct a cifar10 model for HIB.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    model_HIB = nn.Sequential(
        model,
        StochasticLayer(
            embedding_size=embedding_size
        )
    )

    return model_HIB

def Casia_HIB(embedding_size=128):
    """
    Construct a Casia Webface model for HIB.
    """
    # Embedding dimension
    model = resnet50(num_classes=embedding_size)

    model_HIB = nn.Sequential(
        model,
        StochasticLayer(
            embedding_size=embedding_size
        )
    )

    return model_HIB