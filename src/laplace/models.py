#!/usr/bin/env python3

import torch


from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet, SampleNet
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class LaplaceHead(SampleNet):
    def __init__(self, backbone):
        super().__init__()
        self.linear = backbone.linear
        self.convnet = backbone.conv

    def pass_through(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x

    def sample(self, X, samples):
        preds = []

        conv_out = self.convnet(X)

        mu = parameters_to_vector(self.linear.parameters())

        for net_sample in samples:
            vector_to_parameters(net_sample, self.linear.parameters())
            pred = self.linear(conv_out)
            preds.append(pred)

        vector_to_parameters(mu, self.linear.parameters())

        preds = torch.stack(preds, dim=-1)

        return preds.mean(dim=-1), preds.std(dim=-1), preds

    def forward(self, x, use_samples=True):

        if use_samples:
            if hasattr(self, "samples"):
                return self.sample(x, self.samples)
            else:
                raise AttributeError("No samples generated")
        else:
            return self.pass_through(x)

    def generate_nn_samples(self, mu_q, sigma_q, n_samples):
        print(f"Generating {n_samples} samples of model weights")
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


def sample_nn_weights(parameters, posterior_scale, n_samples=100):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    samples = samples + parameters
    return samples
