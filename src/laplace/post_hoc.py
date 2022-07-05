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

    return mu_q, sigma_q


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 25
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
    model.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))

    mu_q, sigma_q = post_hoc(model, train_loader, device)
    torch.save(mu_q.detach().cpu(), "pretrained/laplace/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), "pretrained/laplace/laplace_sigma.pt")
