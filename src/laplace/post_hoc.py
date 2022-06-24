import logging
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
)
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam

from src.hessian.layerwise import ContrastiveHessianCalculator
from src.models.utils import test_model
from src.laplace.utils import (
    sample_nn_weights,
    get_sample_accuracy,
    generate_predictions_from_samples,
)
from src.laplace.miners import AllPermutationsMiner


def run_experiment(
    net: nn.Module,
    train_loader: DataLoader,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device="cpu",
    epochs=20,
    lr=3e-4,
):
    logging.info("Finding MAP solution.")
    train_metric(net, train_loader, epochs, lr, device)

    accuracy = test_model(train_loader, id_loader.dataset, net, device)
    logging.info(f"Accuracy after training is {100*accuracy['precision_at_1']:.2f}%.")

    logging.info("Computing hessian.")
    mu_q, sigma_q = post_hoc(net, train_loader, device)

    logging.info("Sampling.")
    samples = sample_nn_weights(mu_q, sigma_q)

    device = "cpu"
    net = net.to(device)
    samples = samples.to(device)

    accuracies = get_sample_accuracy(
        train_loader.dataset,
        id_loader.dataset,
        net,
        net.linear,
        samples,
        device,
    )
    logging.info(f"Sample accuracies = {accuracies}")

    logging.info("Generating predictions from samples.")
    preds = (
        generate_predictions_from_samples(id_loader, samples, net, net.linear, device)
        .detach()
        .cpu()
    )

    preds_ood = (
        generate_predictions_from_samples(ood_loader, samples, net, net.linear, device)
        .detach()
        .cpu()
    )

    np.save("laplace_mu.npy", preds.mean(dim=0))
    np.save("laplace_sigma_sq.npy", preds.var(dim=0))

    np.save("laplace_mu.npy", preds_ood.mean(dim=0))
    np.save("laplace_sigma_sq.npy", preds_ood.var(dim=0))


def train_metric(
    net: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device="cpu"
):
    """
    Train a metric learning model.
    """
    miner = miners.MultiSimilarityMiner()
    contrastive_loss = losses.ContrastiveLoss()
    optim = Adam(net.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            output = net(x)
            hard_pairs = miner(output, y)
            loss = contrastive_loss(output, y, hard_pairs)
            loss.backward()
            optim.step()


def post_hoc(
    model: nn.Module,
    train_loader: DataLoader,
    device="cpu",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    preinference_model: nn.Module = model.conv
    inference_model: nn.Module = model.linear

    calculator = ContrastiveHessianCalculator()
    calculator.init_model(model.linear)
    compute_hessian = calculator.compute_batch_pairs
    miner = AllPermutationsMiner()
    h = []
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        x_conv = preinference_model(x)  # .detach()
        output = inference_model(x_conv)
        # output = net(x)
        hard_pairs = miner(y)
        hessian = (
            compute_hessian(inference_model, output, x_conv, y, hard_pairs)
            / x.shape[0]
            * len(train_loader.dataset)
        )
        h.append(hessian)
    h = torch.stack(h, dim=0).sum(dim=0).to(device)
    if (h < 0).sum():
        logging.warn("Found negative values in Hessian.")
    h += 1

    mu_q = parameters_to_vector(inference_model.parameters())
    # mu_q = parameters_to_vector(net.parameters())
    sigma_q = 1 / (h + 1e-6)
    return mu_q, sigma_q


# if __name__ == "__main__":
#     logging.getLogger().setLevel(logging.INFO)

#     run_experiment()
