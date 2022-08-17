import logging
from encodings import normalize_encoding
from pickletools import optimize

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.models import resnet50
from tqdm import tqdm

from src.hessian.layerwise import ContrastiveHessianCalculator, FixedContrastiveHessianCalculator
from src.miners import AllCombinationsMiner, AllPermutationsMiner, AllPositiveMiner


def log_det_ratio(hessian, prior_prec):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def scatter(mu_q, prior_precision_diag):
    return (mu_q * prior_precision_diag) @ mu_q


def log_marginal_likelihood(mu_q, hessian, prior_prec):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = -0.5 * (log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec))
    return neg_log_marglik


def optimize_prior_precision(mu_q, hessian, prior_prec, n_steps=100):

    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()

    return prior_prec


def post_hoc(
    model: nn.Module,
    inference_model: nn.Module,
    train_loader: DataLoader,
    margin: float,
    device="cpu",
    method="full",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    if method == "positives":
        calculator = ContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllPositiveMiner()
    elif method == "fixed":
        calculator = FixedContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllCombinationsMiner()
    elif method == "full":
        calculator = ContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllCombinationsMiner()
    else:
        raise ValueError(f"Unknown method: {method}")

    dataset_size = len(train_loader.dataset)

    calculator.init_model(inference_model)
    h = 0
    with torch.no_grad():
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            hard_pairs = miner(output, y)

            # Total number of possible pairs / number of pairs in our batch
            scaler = dataset_size**2 / x.shape[0] ** 2
            hessian = calculator.compute_batch_pairs(hard_pairs)
            h += hessian * scaler

    if (h < 0).sum():
        logging.warning("Found negative values in Hessian.")

    h = torch.maximum(h, torch.tensor(0))

    logging.info(f"{100 * calculator.zeros / calculator.total_pairs:.2f}% of pairs are zero.")
    logging.info(f"{100 * calculator.negatives / calculator.total_pairs:.2f}% of pairs are negative.")

    map_solution = parameters_to_vector(inference_model.parameters())

    scale = 1.0
    prior_prec = 1.0
    prior_prec = optimize_prior_precision(map_solution, h, torch.tensor(prior_prec))
    posterior_precision = h * scale + prior_prec
    posterior_scale = 1.0 / (posterior_precision.sqrt() + 1e-6)

    return map_solution, posterior_scale
