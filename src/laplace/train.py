import logging

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
    epochs = 10
    freq = 3
    nn_samples = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    contrastive_loss = losses.ContrastiveLoss()
    miner = miners.MultiSimilarityMiner()
    hessian_calculator = ContrastiveHessianCalculator()

    latent_dim = 15
    net = ConvNet(latent_dim)
    net_inference = net.linear
    net.to(device)

    hessian_calculator.init_model(net_inference)

    num_params = sum(p.numel() for p in net_inference.parameters())

    optim = Adam(net.parameters(), lr=3e-4)

    batch_size = 16
    train_set = CIFAR10("data/", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    h = 1e10 * torch.ones((num_params,), device=device)

    kl_weight = 0.1

    for epoch in range(epochs):
        print(f"{epoch=}")
        epoch_losses = []
        # train_laplace = epoch % freq == 0
        train_laplace = False
        print(f"{train_laplace=}")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            mu_q = parameters_to_vector(net_inference.parameters())
            sigma_q = 1 / (h + 1e-6)

            kl = compute_kl_term(mu_q, sigma_q)

            sampled_nn = sample_neural_network_wights(mu_q, sigma_q, n_samples=nn_samples)

            con_losses = []
            if train_laplace:
                h = []

            for nn_i in sampled_nn:
                vector_to_parameters(nn_i, net_inference.parameters())

                x_conv = net.conv(x).detach()
                output = net.linear(x_conv)

                hard_pairs = miner(output, y)
                if train_laplace:
                    try:
                        hessian_batch = hessian_calculator.compute_batch_pairs(
                            net_inference, output, x_conv, y, hard_pairs
                        )
                    except Exception:
                        print(f"{nn_i.shape=}")
                        print(f"{x.shape=}")
                        print(f"{x_conv.shape=}")
                        print(f"{output.shape=}")
                        raise Exception

                    # Adjust hessian to the batch size
                    scaler = images_per_class**2 / len(hard_pairs[0])
                    hessian_batch = hessian_batch * scaler

                    h.append(hessian_batch)

                con_loss = contrastive_loss(output, y, hard_pairs)
                con_losses.append(con_loss)

            if train_laplace:
                h = torch.stack(h).mean(dim=0) if len(h) > 1 else h[0]
                # h += 1

            con_loss = torch.stack(con_losses).mean(dim=0)
            loss = con_loss + kl.mean() * kl_weight
            vector_to_parameters(mu_q, net_inference.parameters())

            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())

        loss_mean = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss_mean=} for {epoch=}")

    torch.save(net.state_dict(), f="models/laplace_model.ckpt")
    torch.save(h, f="models/laplace_hessian.ckpt")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    run()
