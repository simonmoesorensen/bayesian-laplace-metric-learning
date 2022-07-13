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

    id_set = CIFAR10("data/", train=False, transform=transforms.ToTensor())
    id_loader = DataLoader(id_set, batch_size, shuffle=False)
    id_label = "cifar10"

    ood_set = SVHN("data/", split="test", transform=transforms.ToTensor())
    ood_loader = DataLoader(ood_set, batch_size, shuffle=False)
    ood_label = "svhn"

    # ood_label = "noise"

    # ood_set = CIFAR100("data/", train=False, transform=transforms.ToTensor())
    # subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # mask = torch.tensor([ood_set[i][1] in subset_classes for i in range(len(ood_set))])
    # indices = torch.arange(len(ood_set))[mask]
    # ood_set = Subset(ood_set, indices)
    # ood_loader = DataLoader(ood_set, batch_size, shuffle=False)
    # ood_label = "cifar100"

    model = ConvNet(latent_dim).to(device)
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)

    t = time.time()
    mean, variance = evaluate_laplace(model, id_loader, device)
    np.save("results/laplace/id_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save("results/laplace/id_laplace_sigma_sq.npy", variance.detach().cpu().numpy())

    t = time.time()
    mean, variance = evaluate_laplace(model, ood_loader, device)
    np.save("results/laplace/ood_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save("results/laplace/ood_laplace_sigma_sq.npy", variance.detach().cpu().numpy())
