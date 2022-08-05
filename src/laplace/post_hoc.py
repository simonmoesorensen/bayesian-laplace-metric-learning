from encodings import normalize_encoding
import logging
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import resnet50
from pytorch_metric_learning import losses

from src.laplace.metric_learning import train_metric
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.miners import AllPermutationsMiner, AllCombinationsMiner, AllPositiveMiner
from src.laplace.utils import sample_nn_weights, get_sample_accuracy
from src import data, models


def post_hoc(
    model: nn.Module,
    inference_model: nn.Module,
    train_loader: DataLoader,
    margin: float,
    device="cpu",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """

    loss_fn = losses.ContrastiveLoss(neg_margin=margin)
    dataset_size = len(train_loader.dataset)

    calculator = ContrastiveHessianCalculator(margin=margin, device=device)
    calculator.init_model(inference_model)
    miner = AllCombinationsMiner()
    h = 0
    grads = []
    mins = []
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        x.requires_grad = True

        output = model(x)
        hard_pairs = miner(output, y)

        loss = loss_fn(output, y, hard_pairs)
        loss.backward()

        grads.append(torch.norm(x.grad).detach().cpu().item())

        # Total number of possible pairs / number of pairs in our batch
        scaler = dataset_size**2 / x.shape[0] ** 2
        hessian = calculator.compute_batch_pairs(hard_pairs)
        mins.append(hessian.min().detach().cpu().item())
        h += hessian * scaler

    if (h < 0).sum():
        logging.warning("Found negative values in Hessian.")

    fig, ax = plt.subplots()
    ax.scatter(grads, mins)
    ax.set(
        xlabel="Gradient norm",
        ylabel="Hessian min",
    )
    fig.savefig("grads.png")

    h = torch.maximum(h, torch.tensor(0))

    # layer_names = [layer.__class__.__name__ for layer in inference_model]
    # params = [sum(p.numel() for p in layer.parameters()) for layer in inference_model]
    # params = [p for p in params if p > 0]
    # cumsum_params = np.array([0] + params)[:-1] + np.array(params) / 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h.detach().cpu().numpy())
    ax.set(xlabel="Layer", ylabel="Log-Hessian")
    # ax.vlines(params, h.detach().cpu().numpy().min() + 1e-6, h.detach().cpu().numpy().max())
    fig.savefig("hessian.png")

    mu_q = parameters_to_vector(inference_model.parameters())
    sigma_q = 1 / (h.sqrt() + 1e-6)

    return mu_q, sigma_q


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 2
    batch_size = 16
    margin = 0.2
    normalize_encoding = False

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", batch_size, 4)
    id_module.prepare_data()
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    # ood_module = data.SVHNDataModule("/work3/s174433/datasets", batch_size, 4)
    # ood_module.prepare_data()
    # ood_module.setup()
    # ood_loader = ood_module.test_dataloader()

    model = models.ConvNet(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)
    # inference_model = nn.Sequential(model.fc)

    model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt", map_location=device))
    # model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict_normalized.pt", map_location=device))

    mu_q, sigma_q = post_hoc(model, inference_model, train_loader, margin, device)
    torch.save(mu_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_sigma.pt")
    # mu_q = torch.load(f"pretrained/post_hoc/{id_label}/laplace_mu.pt", map_location=device)
    # sigma_q = torch.load(f"pretrained/post_hoc/{id_label}/laplace_sigma.pt", map_location=device)

    samples = sample_nn_weights(mu_q, sigma_q)
    accs = get_sample_accuracy(train_loader.dataset, id_loader.dataset, model, inference_model, samples, device)
    accs = {k: [dic[k] for dic in accs] for k in accs[0]}
    for key, val in accs.items():
        print(f"Post-hoc {key}: {val}")
