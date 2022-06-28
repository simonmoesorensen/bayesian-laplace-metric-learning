import logging

import numpy as np
import torch
from pytorch_metric_learning import losses, miners
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from src.models.conv_net import ConvNet
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.miners import AllPermutationsMiner
from src.laplace.utils import (
    generate_predictions_from_samples,
    get_sample_accuracy,
    sample_nn_weights,
)
from src.utils import test_model


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

    np.save("laplace_mu.npy", mu_q)
    np.save("laplace_sigma.npy", sigma_q)

    logging.info("Sampling.")
    samples = sample_nn_weights(mu_q, sigma_q)

    device = "cpu"  # Do we need to do this?
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

    np.save("id_laplace_mu.npy", preds.mean(dim=0))
    np.save("id_laplace_sigma_sq.npy", preds.var(dim=0))

    np.save("ood_laplace_mu.npy", preds_ood.mean(dim=0))
    np.save("ood_laplace_sigma_sq.npy", preds_ood.var(dim=0))


def train_metric(
    net: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device="cpu"
):
    """
    Train a metric learning model.
    """
    # # lambda1 = 1.01
    # lambda1 = 0.34
    # lambda2 = 3.01
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
            # anchors_p, positives, anchors_n, negatives = hard_pairs
            # loss = (
            #     loss
            #     + lambda1 * torch.norm(output[anchors_p], dim=1).sum()
            #     + lambda1 * torch.norm(output[positives], dim=1).sum()
            # )
            # loss = (
            #     loss
            #     + lambda1
            #     * torch.norm(output[anchors_p] + output[positives], dim=1).sum()
            # )
            # distances = torch.norm(output[anchors_n] - output[negatives], dim=1)
            # negative_mask = distances < loss.neg_margin
            # loss = (
            #     loss
            #     + lambda1 * torch.norm(output[anchors_n][negative_mask], dim=1).sum()
            #     + lambda1 * torch.norm(output[negatives][negative_mask], dim=1).sum()
            # )
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 10
    epochs = 20
    lr = 3e-4
    batch_size = 128

    train_set = CIFAR10("data/", train=True, download=True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    id_set = CIFAR10("data/", train=False, download=True)
    id_loader = DataLoader(
        id_set,
        batch_size,
        shuffle=False,
    )

    ood_set = CIFAR100("data/", train=False, download=True)
    subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask = torch.tensor([ood_set[i][1] in subset_classes for i in range(len(ood_set))])
    indices = torch.arange(len(ood_set))[mask]
    ood_set = Subset(ood_set, indices)
    ood_loader = DataLoader(ood_set, batch_size, shuffle=False, num_workers=4)

    model = ConvNet(latent_dim)

    run_experiment(model, train_loader, id_loader, ood_loader, device, epochs, lr)
