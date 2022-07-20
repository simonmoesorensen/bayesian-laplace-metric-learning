import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.models import resnet50

from src.laplace.utils import generate_predictions_from_samples_rolling, sample_nn_weights
from src import data, models


def evaluate_laplace(net, inference_net, loader, mu_q, sigma_q, device="cpu"):
    logging.info("Sampling.")
    samples = sample_nn_weights(mu_q, sigma_q)

    logging.info("Generating predictions from samples.")
    pred_mean, pred_var = generate_predictions_from_samples_rolling(loader, samples, net, inference_net, device)
    return pred_mean, pred_var


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    method = "post_hoc"

    latent_dim = 32
    batch_size = 512
    normalize_encoding = False

    id_module = data.CIFAR10DataModule("data/", batch_size, 4)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    ood_module = data.SVHNDataModule("data/", batch_size, 4)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()

    logging.info("Loading pretrained model.")
    model = models.ConvNet(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)
    # inference_model = model.fc
    model.load_state_dict(torch.load(f"pretrained/{method}/{id_label}/state_dict.pt", map_location=device))

    id_label = id_module.name.lower()
    ood_label = ood_module.name.lower()

    mu_q = torch.load(f"pretrained/{method}/{id_label}/laplace_mu.pt", map_location=device)
    sigma_q = torch.load(f"pretrained/{method}/{id_label}/laplace_sigma.pt", map_location=device)

    mean, variance = evaluate_laplace(model, inference_model, id_loader, mu_q, sigma_q, device)
    np.save(f"results/{method}/{id_label}/id_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy", variance.detach().cpu().numpy())

    mean, variance = evaluate_laplace(model, inference_model, ood_loader, mu_q, sigma_q, device)
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy", mean.detach().cpu().numpy())
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy", variance.detach().cpu().numpy())
