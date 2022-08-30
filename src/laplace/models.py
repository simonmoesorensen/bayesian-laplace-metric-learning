#!/usr/bin/env python3

import torch
from torch import Tensor, nn

from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet, SampleNet
from torch.nn.utils.convert_parameters import vector_to_parameters


class LaplaceHead(SampleNet):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        self.backbone.linear.add_module("l2norm", L2Norm())

    def pass_through(self, x):
        return self.backbone(x)

    def sample(self, X, samples):
        preds = []

        for net_sample in samples:
            vector_to_parameters(net_sample, self.inference_model.parameters())
            pred = self.backbone(X)
            preds.append(pred)

        preds = torch.stack(preds, dim=-1)

        return preds.mean(dim=-1), preds.std(dim=-1)

    def forward(self, x, use_samples=True):
        if use_samples:
            if hasattr(self, "samples"):
                return self.sample(x, self.samples)
            else:
                raise AttributeError("No samples generated")
        else:
            return self.pass_through(x), None

    def generate_nn_samples(self, mu_q, sigma_q, n_samples):
        self.samples = sample_nn_weights(mu_q, sigma_q, n_samples=n_samples)


def FashionMNIST_Laplace(embedding_size=128):
    """
    Construct a mnist model for Laplace.
    """
    model = FashionMNISTConvNet(latent_dim=embedding_size)
    model_sampler = LaplaceHead(model)

    return model_sampler


def CIFAR10_Laplace(embedding_size=128):
    """
    Construct a cifar10 model for Laplace.
    """
    model = CIFAR10ConvNet(latent_dim=embedding_size)
    model_sampler = LaplaceHead(model)

    return model_sampler


class L2Norm(nn.Module):
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


def sample_nn_weights(parameters, posterior_scale, n_samples=16):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return samples
