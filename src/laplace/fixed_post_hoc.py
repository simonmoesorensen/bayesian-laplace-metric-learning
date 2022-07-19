import torch

import gc
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_metric_learning import losses, miners
from torch import batch_norm_stats, negative, nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, RandomSampler, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from tqdm import tqdm
from stochman import nnj

from src.models.conv_net import ConvNet
from src.laplace.hessian.stochman import MseHessianCalculator
from src.laplace.hessian.layerwise import RmseHessianCalculator
from src.laplace.miners import AllPermutationsMiner, AllCombinationsMiner, AllPositiveMiner
from src.laplace.utils import (
    generate_fake_predictions_from_samples,
    generate_predictions_from_samples,
    get_sample_accuracy,
    sample_nn_weights,
)
from src.utils import test_model, get_embedding_indices_within_margin


# def post_hoc(
#     model,
#     train_loader,
#     margin: float,
#     device="cpu",
# ):
#     """
#     Run post-hoc laplace on a pretrained metric learning model.
#     """
#     inference_model = model.linear

#     train_set = train_loader.dataset
#     train_targets = torch.tensor(train_set.targets, device=device)

#     z = []
#     for x, _ in train_loader:
#         z.append(model(x.to(device)))
#     z = torch.concat(z, dim=0)
#     assert z.shape == (50000, 25)
#     # norms = z @ z.T

#     # feature_maps = []

#     # def fw_hook_get_latent(module, input, output):
#     #     feature_maps.append(output.detach())

#     # def fw_hook_get_input(module, input, output):
#     #     feature_maps = [input[0].detach()]

#     # inference_model[0].register_forward_hook(fw_hook_get_input)
#     # for k in range(len(inference_model)):
#     #     inference_model[k].register_forward_hook(fw_hook_get_latent)

#     # images_per_class = 5000

#     calculator = MseHessianCalculator("exact")
#     calculator.init_model(inference_model)

#     calculator_old = RmseHessianCalculator()
#     calculator_old.init_model(inference_model)

#     h = 0
#     for i, (x, y) in enumerate(tqdm(train_set)):
#         y = torch.tensor(y)
#         x = x.unsqueeze(0)
#         x, y = x.to(device), y.to(device)

#         z1 = model(x)
#         hessian_anchor = calculator(inference_model)
#         assert torch.isclose(hessian_anchor, calculator_old.compute_batch(inference_model, 25)).all()

#         positive_indices = y == train_targets
#         positive_indices = torch.where(positive_indices)[0].cpu().numpy()
#         positives = Subset(train_set, positive_indices)
#         for x2, _ in DataLoader(positives, batch_size=32):
#             # # Total number of positive pairs / number of positive pairs in our batch
#             # scaler = images_per_class**2 / len(hard_pairs[0])
#             model(x2.to(device))
#             calculator_old.compute_batch(inference_model, 25)
#             hessian_positive = calculator(inference_model)  # scaler
#             h += hessian_anchor + hessian_positive

#         negative_indices = y != train_targets
#         # in_margin = norms[i, :] < margin
#         in_margin = (z1 @ z.T).squeeze() < margin
#         negatives_in_margin = torch.logical_and(negative_indices, in_margin)
#         negatives_in_margin = torch.where(negatives_in_margin)[0].cpu().numpy()
#         # negative_indices = torch.where(y != train_targets)[0]
#         print(len(negatives_in_margin))
#         negatives = Subset(train_set, negatives_in_margin)
#         for x2, _ in DataLoader(negatives, batch_size=256):
#             # scaler = (images_per_class*())**2 / len(hard_pairs[0])
#             model(x2.to(device))
#             hessian_negative = calculator.compute_batch(inference_model, 25)  # scaler
#             h += -(hessian_anchor + hessian_negative)

#     if (h < 0).sum():
#         logging.warn("Found negative values in Hessian.")

#     mu_q = parameters_to_vector(inference_model.parameters())
#     # mu_q = parameters_to_vector(net.parameters())
#     sigma_q = 1 / (h + 1e-6)

#     return mu_q, sigma_q


def post_hoc(
    model,
    train_loader,
    margin: float,
    device="cpu",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    inference_model = model.linear

    images_per_class = 5000
    n_classes = 10

    calculator = MseHessianCalculator("exact")
    calculator.init_model(inference_model)

    train_set = train_loader.dataset
    train_targets = torch.tensor(train_set.targets)

    h = 0
    for c in range(n_classes):
        class_indices = train_targets == c
        class_loader = DataLoader(Subset(train_set, class_indices), batch_size=train_loader.batch_size)
        for x1, _ in class_loader:
            x1 = x1.to(device)
            other_loader = DataLoader(
                train_set, batch_size=train_loader.batch_size, sampler=RandomSampler(train_set, num_samples=64)
            )
            # z1 = model(x1)
            for x2, y2 in other_loader:
                x2 = x2.to(device)
                y2 = y2.to(device)
                # z2 = model(x2)
                # p = torch.stack([torch.where(y1i == y2)[0] for y1i in y1], dim=0)
                # n = torch.stack([torch.where(y1i != y2)[0] for y1i in y1], dim=0)
                p = torch.where(c == y2)[0]
                n = torch.where(c != y2)[0]

                # pos_scaler = images_per_class**2 / len(ap)
                # neg_scaler = images_per_class * (1 - n_classes) ** 2 / len(an)
                model(x1)
                anchor_hessian = calculator(inference_model)

                if p.shape[0] > 0:
                    h += p.shape[0] * anchor_hessian
                    model(x2[p])
                    h += p.shape[0] * calculator(inference_model)

                if n.shape[0] > 0:
                    h -= n.shape[0] * anchor_hessian
                    model(x2[n])
                    h -= n.shape[0] * calculator(inference_model)

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
    model.linear = nnj.Sequential(
        nnj.Linear(6272, 64),
        nnj.Tanh(),
        nnj.Linear(64, latent_dim),
    ).to(device)
    model.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))

    mu_q, sigma_q = post_hoc(model, train_loader, 0.2, device)
    torch.save(mu_q.detach().cpu(), "pretrained/laplace/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), "pretrained/laplace/laplace_sigma.pt")
