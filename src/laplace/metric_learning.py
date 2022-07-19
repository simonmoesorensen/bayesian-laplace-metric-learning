import logging

import torch
from pytorch_metric_learning import losses, miners
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from src.laplace.utils import test_model
from src.models.conv_net import ConvNet
from src import data


def train_metric(net: nn.Module, train_loader: DataLoader, epochs: int, lr: float, margin: float, device="cpu"):
    """
    Train a metric learning model.
    """
    miner = miners.MultiSimilarityMiner()
    contrastive_loss = losses.ContrastiveLoss(neg_margin=margin)
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

    latent_dim = 32
    epochs = 15
    lr = 3e-4
    batch_size = 128
    margin = 0.2

    id_module = data.CIFAR10DataModule("data/", batch_size, 4)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    model = ConvNet(latent_dim).to(device)
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)

    logging.info("Finding MAP solution.")
    train_metric(model, train_loader, epochs, lr, margin, device)
    torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict.pt")

    k = 10
    results = test_model(train_loader.dataset, id_loader.dataset, model, device, k=k)
    logging.info(f"MAP MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"MAP Accuracy: {100*results['precision_at_1']:.2f}%")
