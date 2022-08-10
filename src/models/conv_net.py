from torch import nn
from src.layers import L2Normalize


class ConvNet(nn.Module):
    def __init__(self, latent_dim=32, normalize=False):
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
            # nn.Linear(6272, latent_dim),
            nn.Linear(6272, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim),
            # nn.Linear(6272, 128),
            # nn.Tanh(),
            # nn.Linear(128, latent_dim),
        ]

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(32, 64, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(64, 128, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(0.25),
        #     nn.Flatten(),
        # )
        # linear_layers = [
        #     # nn.Linear(25088, 4096), # 
        #     # nn.Tanh(), # 
        #     # nn.Linear(4096, 1024), # 
        #     # nn.Tanh(), # 
        #     # nn.Linear(1024, 512), # 
        #     # nn.Tanh(), # 
        #     nn.Linear(512, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     # nn.Tanh(),
        #     # nn.Linear(128, latent_dim),
        # ]

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, stride=1, padding=1),
        #     nn.Tanh(),
        #     nn.Conv2d(16, 32, 3, stride=1, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.Tanh(),
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1),
        #     nn.Tanh(),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.Tanh(),
        #     nn.Flatten(),
        # )
        # linear_layers = [
        #     nn.Linear(8 * 8 * 64, latent_dim),
        # ]

        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class FashionMNISTConvNet(nn.Module):
    def __init__(self, latent_dim=32, normalize=False):
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
            # nn.Linear(4608, 64),  # 4608 6272
            # nn.Tanh(),
            # nn.Linear(64, latent_dim),
            nn.Linear(4608, latent_dim),
            # nn.Linear(4608, 512),
            # nn.Tanh(),
            # nn.Linear(512, latent_dim),
        ]

        # # Frederik's FashionMNIST
        # self.conv = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(784, 512),
        #     nn.Dropout(p=0.25),
        #     nn.Tanh(),
        # )
        # linear_layers = [
        #     nn.Linear(512, 256),
        #     # nn.Dropout(p=0.25),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     # nn.Dropout(p=0.25),
        #     nn.Tanh(),
        #     nn.Linear(128, latent_dim),
        # ]

        # # Frederik's MNIST
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
        #     nn.MaxPool2d(2),
        #     nn.Tanh(),
        #     nn.Conv2d(8, 12, 3, stride=1, padding=1, bias=False),
        #     nn.MaxPool2d(2),
        #     nn.Tanh(),
        #     nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=False),
        #     nn.Tanh(),
        #     nn.Flatten(),
        # )
        # linear_layers = [
        #     nn.Linear(7 * 7 * 12, latent_dim),
        # ]
        
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
