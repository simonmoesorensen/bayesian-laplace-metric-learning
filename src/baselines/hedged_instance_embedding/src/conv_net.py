from torch import nn


class ConvNet(nn.Module):
    def __init__(self, latent_dim=128, n_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 2, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
    

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x