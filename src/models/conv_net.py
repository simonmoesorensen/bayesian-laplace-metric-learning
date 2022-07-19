import torch
from torch import nn, Tensor
from src.laplace.layers import Norm2, Reciprocal, Sqrt


# class L2Normalize(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.normalizer = nn.Sequential(
#             Norm2(dim=1),
#             Sqrt(),
#             Reciprocal(),
#         )

#     def forward(self, x):
#         return x * self.normalizer(x)


class L2Normalize(nn.Module):
    """L2 normalization layer"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, d = x.shape

        norm = torch.norm(x, p=2, dim=1)

        out = torch.einsum("bi,bj->bij", x, x)
        out = torch.einsum("b,bij->bij", 1 / (norm**3 + 1e-6), out)
        out = (
            torch.einsum("b,bij->bij", 1 / (norm + 1e-6), torch.diag(torch.ones(d, device=x.device)).expand(b, d, d))
            - out
        )

        return out


class ConvNet(nn.Module):
    def __init__(self, latent_dim=10, normalize=False):
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
            nn.Linear(6272, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        ]
        if normalize:
            linear_layers.append(L2Normalize())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
