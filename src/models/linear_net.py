from torch import nn

from src.layers import L2Normalize


class LinearNet(nn.Module):
    def __init__(self, latent_dim=32, normalize=False):
        super().__init__()
        linear_layers = [
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, latent_dim),
        ]
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.linear(x)
        return x
