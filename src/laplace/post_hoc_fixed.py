import torch
from laplace.hessian.layerwise import RmseHessianCalculator
from laplace.miners import AllPositiveMiner

import gc
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_metric_learning import losses, miners
from torch import negative, nn
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
    model,
    train_loader,
    device="cpu",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    preinference_model = model.conv
    inference_model = model.linear

    feature_maps = []

    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(model)):
        model[k].register_forward_hook(fw_hook_get_latent)

    # images_per_class = 6000

    calculator = RmseHessianCalculator()
    calculator.init_model(model.linear)
    miner = AllPositiveMiner()
    h = 0
    for x, y in tqdm(train_loader.dataset):
        x, y = x.to(device), y.to(device)

        # TODO: is this right?
        # x_conv = preinference_model(x)  # .detach()
        # output = inference_model(x_conv)
        x1 = model(x)
        positives = Subset(train_loader.dataset, find_all_positives(y, train_loader.dataset))

        for x2 in DataLoader(positives, batch_size=32):
            # # Total number of positive pairs / number of positive pairs in our batch
            # scaler = images_per_class**2 / len(hard_pairs[0])
            hessian = calculator.compute_batch(
                inference_model,
                feature_maps,
                x1,
                x2,
            )  # * scaler
            feature_maps = []
            h += hessian

    if (h < 0).sum():
        logging.warn("Found negative values in Hessian.")

    mu_q = parameters_to_vector(inference_model.parameters())
    # mu_q = parameters_to_vector(net.parameters())
    sigma_q = 1 / (h + 1e-6)

    return mu_q, sigma_q
