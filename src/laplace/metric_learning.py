import logging

import torch
from pytorch_metric_learning import losses, miners
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from tqdm import tqdm

from src.models.conv_net import ConvNet


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


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 25
    epochs = 20
    lr = 3e-4
    batch_size = 16

    train_set = CIFAR10("data/", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    test_set = CIFAR10("data/", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = ConvNet(latent_dim).to(device)

    logging.info("Finding MAP solution.")
    train_metric(model, train_loader, epochs, lr, device)
    torch.save(model.state_dict(), "pretrained/laplace/state_dict.pt")
