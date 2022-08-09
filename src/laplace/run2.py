import logging
import os
import time
from encodings import normalize_encoding
from typing import List
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.models import resnet50
from torchvision.utils import save_image
from tqdm import tqdm

from src import data, models
from src.laplace.evaluate import evaluate_laplace
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.metric_learning import train_metric
from src.laplace.miners import (AllCombinationsMiner, AllPermutationsMiner,
                                AllPositiveMiner)
from src.laplace.post_hoc import post_hoc
from src.laplace.utils import (generate_predictions_from_samples_rolling,
                               get_sample_accuracy, sample_nn_weights,
                               test_model)
from src.visualization.plot_ood import plot_ood
from src.visualization.plot_roc import compute_and_plot_roc_curves

sns.set_theme(style="ticks")

def run_posthoc(latent_dim: int):
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 30
    lr = 3e-4
    batch_size = 128
    margin = 0.2
    normalize_encoding = False

    method = "post_hoc"

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", batch_size)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    model = models.ConvNet(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)

    logging.info("Finding MAP solution.")
    train_metric(model, train_loader, epochs, lr, margin, device)
    torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict.pt")
    # # torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict_normalized.pt")
    # model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt"))

    k = 10
    results = test_model(train_loader.dataset, id_loader.dataset, model, device, k=k)
    logging.info(f"MAP MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"MAP Accuracy: {100*results['precision_at_1']:.2f}%")

    batch_size = 16

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", batch_size, 4)
    id_module.prepare_data()
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    # model = models.ConvNet(latent_dim, normalize_encoding).to(device)
    # inference_model = model.linear
    # model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt", map_location=device))

    # model.eval()

    mu_q, sigma_q = post_hoc(model, inference_model, train_loader, margin, device)
    torch.save(mu_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_sigma.pt")

    batch_size = 512

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", batch_size, 4)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_label = id_module.name.lower()

    ood_module = data.MNISTDataModule("/work3/s174433/datasets", batch_size, 4)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()

    model.load_state_dict(torch.load(f"pretrained/{method}/{id_label}/state_dict.pt", map_location=device))

    id_label = id_module.name.lower()
    ood_label = ood_module.name.lower()

    mu_q = torch.load(f"pretrained/{method}/{id_label}/laplace_mu.pt", map_location=device)
    sigma_q = torch.load(f"pretrained/{method}/{id_label}/laplace_sigma.pt", map_location=device)

    # model.eval()

    mean_id, variance_id = evaluate_laplace(model, inference_model, id_loader, mu_q, sigma_q, device)
    mean_id = mean_id.detach().cpu()
    variance_id = variance_id.detach().cpu()
    np.save(f"results/{method}/{id_label}/id_laplace_mu.npy", mean_id.numpy())
    np.save(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy", variance_id.numpy())

    mean_ood, variance_ood = evaluate_laplace(model, inference_model, ood_loader, mu_q, sigma_q, device)
    mean_ood = mean_ood.detach().cpu()
    variance_ood = variance_ood.detach().cpu()
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy", mean_ood.numpy())
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy", variance_ood.numpy())

    id_title = "FashionMNIST"
    # id_title = "MNIST"
    # id_title = "CIFAR-10"
    id_label = id_title.lower()

    method = "post_hoc"

    ood_title = "MNIST"
    # ood_title = "FashionMNIST"
    # ood_title = "SVHN"
    # ood_title = "CIFAR-100"
    ood_label = ood_title.lower()

    # mu_id = np.load(f"results/{method}/{id_label}/id_laplace_mu.npy")
    # var_id = np.load(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy")
    # mu_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy")
    # var_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy")

    mu_id = mean_id.numpy()
    var_id = variance_id.numpy()
    mu_ood = mean_ood.numpy()
    var_ood = variance_ood.numpy()

    fig, ax = plot_ood(mu_id, var_id, mu_ood, var_ood)
    fig.suptitle(f"Trained on {id_title}, OOD {ood_title}")
    fig.tight_layout()
    fig.savefig(f"results/{method}/{id_label}/{ood_label}/ood_plot.png")

    metrics = compute_and_plot_roc_curves(f"results/{method}/{id_label}/{ood_label}/", var_id, var_ood)
    print(metrics)
    # metrics = pd.DataFrame.from_dict({metric: [val] for metric, val in metrics.items()})
    # metrics_path = f"results/{method}/{id_label}/{ood_label}/ood_metrics.csv"
    # metrics.to_csv(metrics_path, index=False, header=True)

    return metrics["auroc"]


if __name__ == "__main__":

    run_posthoc(2)

    # auroc = {}
    # for d in range(2, 32+1, 2):
    #     tmp = []
    #     for i in range(3):
    #         tmp.append(run_posthoc(d))
    #     auroc[d] = (np.mean(tmp), np.std(tmp))

    # print(auroc)
    # with open("auroc.pkl", "wb") as f:
    #     pickle.dump(auroc, f)

    # df = pd.DataFrame.from_dict(auroc, orient="index").reset_index()
    # df.columns = ["d", "auroc_mean", "auroc_std"]
    # print(df)

    # fig, ax = plt.subplots()
    # ax.errorbar(df["d"], df["auroc_mean"], yerr=df["auroc_std"], fmt="o")
    # ax.set_xlabel("Latent dimensions")
    # ax.set_ylabel("AUROC")
    # fig.tight_layout()
    # fig.savefig("auroc.png")
