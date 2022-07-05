import gc
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_metric_learning import losses, miners
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from tqdm import tqdm

from src.models.conv_net import ConvNet
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.miners import AllPermutationsMiner, AllCombinationsMiner, AllPositiveMiner
from src.laplace.utils import (
    generate_fake_predictions_from_samples,
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
    train=True,
):
    net = net.to(device)

    if train:
        logging.info("Finding MAP solution.")
        train_metric(net, train_loader, epochs, lr, device)
        torch.save(net.state_dict(), "pretrained/laplace/state_dict.pt")

        accuracy = test_model(train_loader.dataset, id_loader.dataset, net, device)
        logging.info(f"Accuracy after training is {100*accuracy['precision_at_1']:.2f}%.")

        logging.info("Computing hessian.")
        mu_q, sigma_q = post_hoc(net, train_loader, device)

        torch.save(mu_q.detach().cpu(), "pretrained/laplace/laplace_mu.pt")
        torch.save(sigma_q.detach().cpu(), "pretrained/laplace/laplace_sigma.pt")

        return

    logging.info("Loading pretrained model.")
    net.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))

    accuracy = test_model(train_loader.dataset, id_loader.dataset, net, device)
    logging.info(f"Accuracy after training is {100*accuracy['precision_at_1']:.2f}%.")

    mu_q = torch.load("pretrained/laplace/laplace_mu.pt", map_location=device)
    sigma_q = torch.load("pretrained/laplace/laplace_sigma.pt", map_location=device)

    logging.info("Sampling.")
    samples = sample_nn_weights(mu_q, sigma_q)

    # sample_accuracies = get_sample_accuracy(train_loader.dataset, id_loader.dataset, net, net.linear, samples, device)
    # logging.info(f"Sample accuracy mean={np.mean(sample_accuracies)} var={np.var(sample_accuracies)}")

    device = "cpu"  # TODO: do we need to do this?
    net = net.to(device)
    samples = samples.to(device)

    logging.info("Generating predictions from ID samples.")
    preds = generate_predictions_from_samples(id_loader, samples, net, net.linear, device).detach().cpu()
    np.save(
        "results/laplace/id_laplace_mu.npy",
        preds.numpy().mean(axis=0),
    )
    np.save(
        "results/laplace/id_laplace_sigma_sq.npy",
        preds.numpy().var(axis=0),
    )

    logging.info("Generating predictions from OOD samples.")
    # preds_ood = generate_predictions_from_samples(ood_loader, samples, net, net.linear, device).detach().cpu()
    preds_ood = generate_fake_predictions_from_samples(id_loader, samples, net, net.linear, device).detach().cpu()
    np.save(
        "results/laplace/ood_laplace_mu.npy",
        preds_ood.numpy().mean(axis=0),
    )
    np.save(
        "results/laplace/ood_laplace_sigma_sq.npy",
        preds_ood.numpy().var(axis=0),
    )


def train_metric(net: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device="cpu"):
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

    images_per_class = 6000

    calculator = ContrastiveHessianCalculator()
    calculator.init_model(model.linear)
    compute_hessian = calculator.compute_batch_pairs
    miner = AllPositiveMiner()
    h = []
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        # TODO: is this right?
        x_conv = preinference_model(x)  # .detach()
        output = inference_model(x_conv)
        # output = output * model.normalizer(output)
        # output = net(x)

        hard_pairs = miner(output, y)
        assert len(hard_pairs[3]) == 0  # All positive miner, no negative elements

        # Total number of positive pairs / number of positive pairs in our batch
        scaler = images_per_class**2 / len(hard_pairs[0])
        hessian = compute_hessian(inference_model, output, x_conv, y, hard_pairs) * scaler
        h.append(hessian)
    h = torch.stack(h, dim=0).sum(dim=0).to(device)
    if (h < 0).sum():
        logging.warn("Found negative values in Hessian.")

    mu_q = parameters_to_vector(inference_model.parameters())
    # mu_q = parameters_to_vector(net.parameters())
    sigma_q = 1 / (h + 1e-6)

    calculator.clean_up()
    del calculator
    gc.collect()

    return mu_q, sigma_q


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 2
    epochs = 20
    lr = 3e-4
    batch_size = 16

    train_set = CIFAR10("data/", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    id_set = CIFAR10("data/", train=False, transform=transforms.ToTensor())
    id_loader = DataLoader(id_set, batch_size, shuffle=False)

    # ood_set = CIFAR100("data/", train=False, transform=transforms.ToTensor())
    # subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # mask = torch.tensor([ood_set[i][1] in subset_classes for i in range(len(ood_set))])
    # indices = torch.arange(len(ood_set))[mask]
    # ood_set = Subset(ood_set, indices)
    # ood_loader = DataLoader(ood_set, batch_size, shuffle=False)

    ood_set = SVHN("data/", split="test", transform=transforms.ToTensor())
    ood_loader = DataLoader(ood_set, batch_size, shuffle=False)

    model = ConvNet(latent_dim).to(device)

    run_experiment(model, train_loader, id_loader, ood_loader, device, epochs, lr, train=False)
