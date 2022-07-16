import logging

import torch
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import resnet50
from laplace.metric_learning import train_metric

from src.models.conv_net import ConvNet
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.miners import AllPermutationsMiner, AllCombinationsMiner, AllPositiveMiner
from src import data


def post_hoc(
    model: nn.Module,
    train_loader: DataLoader,
    margin: float,
    device="cpu",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    preinference_model: nn.Module = model.conv
    inference_model: nn.Module = model.linear

    images_per_class = 5000

    calculator = ContrastiveHessianCalculator(margin=margin)
    calculator.init_model(model.linear)
    compute_hessian = calculator.compute_batch_pairs
    miner = AllCombinationsMiner()
    h = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        # TODO: is this right?
        x_conv = preinference_model(x)  # .detach()
        output = inference_model(x_conv)
        # output = output * model.normalizer(output)
        # output = net(x)

        hard_pairs = miner(output, y)
        # assert len(hard_pairs[3]) == 0  # All positive miner, no negative elements

        # Total number of positive pairs / number of positive pairs in our batch
        scaler = images_per_class**2 / len(hard_pairs[0])
        h += compute_hessian(inference_model, output, x_conv, y, hard_pairs) * scaler

    if (h < 0).sum():
        logging.warn("Found negative values in Hessian.")

    mu_q = parameters_to_vector(inference_model.parameters())
    sigma_q = 1 / (h + 1e-6)

    return mu_q, sigma_q


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 25
    batch_size = 16
    epochs = 30
    lr = 3e-4
    margin = 0.2

    id_module = data.CIFAR10DataModule("data/", batch_size, 4)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()

    ood_module = data.SVHNDataModule("data/", batch_size, 4)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()

    model = ConvNet(latent_dim).to(device)
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)

    # model.load_state_dict(torch.load("pretrained/laplace/state_dict.pt", map_location=device))
    train_metric(model, train_loader, epochs, lr, margin, device)

    mu_q, sigma_q = post_hoc(model, train_loader, margin, device)
    torch.save(mu_q.detach().cpu(), "pretrained/laplace/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), "pretrained/laplace/laplace_sigma.pt")
