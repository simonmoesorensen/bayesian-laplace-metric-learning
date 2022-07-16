import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.models import resnet50

from src.laplace.utils import generate_predictions_from_samples_rolling, sample_nn_weights
from src.models.conv_net import ConvNet
from src import data


def evaluate_laplace(net, loader, device="cpu"):
    logging.info("Loading pretrained model.")
    net.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))

    mu_q = torch.load("pretrained/laplace/laplace_mu.pt", map_location=device)
    sigma_q = torch.load("pretrained/laplace/laplace_sigma.pt", map_location=device)

    logging.info("Sampling.")
    samples = sample_nn_weights(mu_q, sigma_q)

    logging.info("Generating predictions from samples.")
    pred_mean, pred_var = generate_predictions_from_samples_rolling(loader, samples, net, net.linear, device)
    return pred_mean, pred_var


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 25
    batch_size = 512

    id_module = data.CIFAR10DataModule("data/", batch_size, 4)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()

    ood_module = data.SVHNDataModule("data/", batch_size, 4)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()

    model = ConvNet(latent_dim).to(device)

    id_label = id_module.name.lower()
    ood_label = ood_module.name.lower()

    mean, variance = evaluate_laplace(model, id_loader, device)
    np.save(f"results/laplace/{id_label}/id_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save(f"results/laplace/{id_label}/id_laplace_sigma_sq.npy", variance.detach().cpu().numpy())

    mean, variance = evaluate_laplace(model, ood_loader, device)
    np.save(f"results/laplace/{id_label}/{ood_label}/ood_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save(f"results/laplace/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy", variance.detach().cpu().numpy())
