#!/usr/bin/env python3

import torch
from torch import Tensor, nn


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
            torch.einsum(
                "b,bij->bij",
                1 / (norm + 1e-6),
                torch.diag(torch.ones(d, device=x.device)).expand(b, d, d),
            )
            - out
        )

        return out
