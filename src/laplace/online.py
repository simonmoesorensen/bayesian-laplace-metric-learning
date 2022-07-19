import logging
from tkinter import N

import torch
from tqdm import tqdm
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.models.conv_net import ConvNet
from src import data
from src.laplace.utils import test_model


def compute_kl_term(mu_q, sigma_q):
    """
    https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    """
    k = len(mu_q)
    return 0.5 * (-torch.log(sigma_q) - k + torch.dot(mu_q, mu_q) + torch.sum(sigma_q))


def sample_neural_network_wights(parameters, posterior_scale, n_samples=32):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def run():
    epochs = 30
    freq = 3
    nn_samples = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    lr = 3e-4
    latent_dim = 25
    margin = 0.2

    train_module = data.CIFAR10DataModule("data/", batch_size, 4)
    train_module.setup()
    train_loader = train_module.train_dataloader()
    id_loader = train_module.test_dataloader()
    id_label = train_module.name.lower()

    net = ConvNet(latent_dim)
    net_inference = net.linear
    net.to(device)

    mu_q, sigma_q = train_online(epochs, freq, nn_samples, device, lr, train_loader, margin, net, net_inference)
    torch.save(net.state_dict(), f"pretrained/online/{id_label}/state_dict.pt")
    torch.save(mu_q.detach().cpu(), f"pretrained/online/{id_label}/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), f"pretrained/online/{id_label}/laplace_sigma.pt")

    # net.load_state_dict(torch.load(f"pretrained/online/{id_label}/state_dict.pt"))

    k = 10
    results = test_model(train_loader.dataset, id_loader.dataset, net, device, k=k)
    logging.info(f"MAP MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"MAP Accuracy: {100*results['precision_at_1']:.2f}%")


def train_online(epochs, freq, nn_samples, device, lr, train_loader, margin, net, net_inference):
    contrastive_loss = losses.ContrastiveLoss(neg_margin=margin)
    miner = miners.MultiSimilarityMiner()
    hessian_calculator = ContrastiveHessianCalculator(margin=margin, device=device)

    hessian_calculator.init_model(net_inference)

    num_params = sum(p.numel() for p in net_inference.parameters())

    optim = Adam(net.parameters(), lr=lr)

    images_per_class = 5000

    h = 1e10 * torch.ones((num_params,), device=device)

    # kl_weight = 0.1

    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        compute_hessian = epoch % freq == 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            mu_q = parameters_to_vector(net_inference.parameters())
            sigma_q = 1 / (h + 1e-6)

            kl = compute_kl_term(mu_q, sigma_q)

            sampled_nn = sample_neural_network_wights(mu_q, sigma_q, n_samples=nn_samples)

            con_loss = 0
            if compute_hessian:
                h = 0

            for nn_i in sampled_nn:
                vector_to_parameters(nn_i, net_inference.parameters())
                output = net(x)
                hard_pairs = miner(output, y)

                if compute_hessian:
                    # Adjust hessian to the batch size
                    scaler = images_per_class**2 / len(hard_pairs[0])
                    hessian_batch = hessian_calculator.compute_batch_pairs(net_inference, hard_pairs)
                    h += hessian_batch * scaler

                con_loss += contrastive_loss(output, y, hard_pairs)

            if compute_hessian:
                h /= nn_samples

            con_loss /= nn_samples
            loss = con_loss  # + kl.mean() * kl_weight
            vector_to_parameters(mu_q, net_inference.parameters())

            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())

        loss_mean = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss_mean=} for {epoch=}")

    mu_q = parameters_to_vector(net_inference.parameters())
    sigma_q = 1 / (h + 1e-6)
    return mu_q, sigma_q


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    run()
