import torch
from torch import nn
from torchvision.models import resnet50

from src.layers import L2Normalize
from src.baselines.Backbone.models import MNIST_Backbone

class MNIST(nn.Module):
    def __init__(self, embedding_size=32, normalize=False):
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
            nn.Linear(4608, embedding_size),
        ]
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

# def MNIST(embedding_size=128):
#     """
#     Construct a mnist model for Backbone.
#     """
#     # Embedding dimension
#     model = resnet50(num_classes=embedding_size)

#     # Adapt to 1 channel inputs
#     model.conv1 = nn.Conv2d(
#         1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     )

#     return model

class CIFAR10(nn.Module):
    def __init__(self, embedding_size=32, normalize=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
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
            nn.Linear(256, embedding_size),
        ]
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(0.25),
        #     nn.Flatten(),
        # )
        # linear_layers = [
        #     nn.Linear(512, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, embedding_size),
        # ]

        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class Casia(nn.Module):
    def __init__(self, embedding_size=32, normalize=False):
        super().__init__()
        
        n_channels = 3
        encoder_hid = 128
        filter_size = 5
        pad = filter_size // 2

        self.conv = nn.Sequential(  # (bs, 3, 64, 64)
            nn.Conv2d(n_channels, encoder_hid, filter_size, padding=pad),
            nn.Tanh(),  # (bs, hid, 64, 64)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 32, 32)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 16, 16)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 8, 8)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 4, 4),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 2, 2),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad),
            nn.MaxPool2d(2),
            nn.Tanh(),  # (bs, hid, 1, 1),
            nn.Flatten(),  # (bs, hid*1*1)
        )
        linear_layers = [
            nn.Linear(encoder_hid, embedding_size),
        ]

        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

