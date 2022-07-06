import gc
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms

from src.laplace.utils import (
    generate_fake_predictions_from_samples,
    generate_predictions_from_samples,
    generate_predictions_from_samples_rolling,
    get_sample_accuracy,
    sample_nn_weights,
)
from src.models.conv_net import ConvNet
from src.utils import test_model


def evaluate_laplace(net, train_loader, id_loader, ood_loader, device="cpu"):
    logging.info("Loading pretrained model.")
    net.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))

    # accuracy = test_model(train_loader.dataset, id_loader.dataset, net, device)
    # logging.info(f"Accuracy after training is {100*accuracy['precision_at_1']:.2f}%.")

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
    preds_ood = generate_predictions_from_samples(ood_loader, samples, net, net.linear, device).detach().cpu()
    # preds_ood = generate_fake_predictions_from_samples(id_loader, samples, net, net.linear, device).detach().cpu()
    np.save(
        "results/laplace/ood_laplace_mu.npy",
        preds_ood.numpy().mean(axis=0),
    )
    np.save(
        "results/laplace/ood_laplace_sigma_sq.npy",
        preds_ood.numpy().var(axis=0),
    )

    # logging.info("Generating predictions from ID samples.")
    # pred_mean, pred_var = generate_predictions_from_samples(id_loader, samples, net, net.linear, device)
    # np.save("results/laplace/id_laplace_mu.npy", pred_mean.detach().cpu().numpy())
    # np.save("results/laplace/id_laplace_sigma_sq.npy", pred_var.detach().cpu().numpy())

    # logging.info("Generating predictions from OOD samples.")
    # pred_ood_mean, pred_ood_var = generate_predictions_from_samples(ood_loader, samples, net, net.linear, device)
    # # preds_ood = generate_fake_predictions_from_samples(id_loader, samples, net, net.linear, device).detach().cpu()
    # np.save("results/laplace/ood_laplace_mu.npy", pred_ood_mean.detach().cpu().numpy())
    # np.save("results/laplace/ood_laplace_sigma_sq.npy", pred_ood_var.detach().cpu().numpy())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 25
    batch_size = 16

    train_set = CIFAR10("data/", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    id_set = CIFAR10("data/", train=False, transform=transforms.ToTensor())
    id_loader = DataLoader(id_set, batch_size, shuffle=False)

    ood_set = SVHN("data/", split="test", transform=transforms.ToTensor())
    ood_loader = DataLoader(ood_set, batch_size, shuffle=False)

    model = ConvNet(latent_dim).to(device)

    evaluate_laplace(model, train_loader, id_loader, ood_loader, device)
