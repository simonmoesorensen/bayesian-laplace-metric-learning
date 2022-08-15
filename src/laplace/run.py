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
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.models import resnet50
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_lightning import LightningDataModule

from src import data, models
from src.laplace.evaluate import evaluate_laplace
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.laplace.metric_learning import train_metric
from src.laplace.miners import (AllCombinationsMiner, AllPermutationsMiner,
                                AllPositiveMiner)
from src.laplace.post_hoc import post_hoc
from src.laplace.utils import (generate_predictions_from_samples_rolling,
                               get_sample_accuracy, sample_nn_weights,
                               test_model, test_model_expected_distance)
from src.visualization.plot_ood import plot_ood
from src.visualization.plot_roc import compute_and_plot_roc_curves
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.distances import LpDistance

sns.set_theme(style="ticks")

def run_posthoc(latent_dim: int, module_id, module_ood, model_module):
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 30
    lr = 3e-4
    batch_size = 16
    margin = 0.2
    normalize_encoding = False

    method = "post_hoc"

    id_module = module_id("/work3/s174433/datasets", batch_size)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()
    id_title = id_module.name
    id_label = id_title.lower()

    ood_module = module_ood("/work3/s174433/datasets", batch_size)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()
    ood_title = ood_module.name
    ood_label = ood_title.lower()

    # batch_size = 128
    # id_module = module_id("/work3/s174433/datasets", batch_size)
    # id_module.setup()
    # train_loader = id_module.train_dataloader()
    # id_loader = id_module.test_dataloader()
    # id_label = id_module.name.lower()

    model = model_module(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear

    logging.info("Finding MAP solution.")
    train_metric(model, train_loader, epochs, lr, margin, device)
    torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict.pt")
    # model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt"))

    k = 5
    results = test_model(train_loader.dataset, id_loader.dataset, model, device, k=k)
    logging.info(f"MAP MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"MAP Accuracy: {100*results['precision_at_1']:.2f}%")

    # batch_size = 16
    # id_module = module_id("/work3/s174433/datasets", batch_size)
    # id_module.setup()
    # train_loader = id_module.train_dataloader()
    # id_loader = id_module.test_dataloader()
    # id_label = id_module.name.lower()

    model = model_module(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear
    model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt", map_location=device))

    mu_q, sigma_q = post_hoc(model, inference_model, train_loader, margin, device)
    torch.save(mu_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_mu.pt")
    torch.save(sigma_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_sigma.pt")

    # batch_size = 512
    # id_module = module_id("/work3/s174433/datasets", batch_size)
    # id_module.setup()
    # train_loader = id_module.train_dataloader()
    # id_loader = id_module.test_dataloader()
    # id_label = id_module.name.lower()

    # ood_module = module_ood("/work3/s174433/datasets", batch_size)
    # ood_module.setup()
    # ood_loader = ood_module.test_dataloader()
    # ood_label = ood_module.name.lower()

    model.load_state_dict(torch.load(f"pretrained/{method}/{id_label}/state_dict.pt", map_location=device))

    mu_q = torch.load(f"pretrained/{method}/{id_label}/laplace_mu.pt", map_location=device)
    sigma_q = torch.load(f"pretrained/{method}/{id_label}/laplace_sigma.pt", map_location=device)

    mu_id, var_id = evaluate_laplace(model, inference_model, id_loader, mu_q, sigma_q, device)
    mu_id = mu_id.detach().cpu().numpy()
    var_id = var_id.detach().cpu().numpy()
    np.save(f"results/{method}/{id_label}/id_laplace_mu.npy", mu_id)
    np.save(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy", var_id)

    mu_ood, var_ood = evaluate_laplace(model, inference_model, ood_loader, mu_q, sigma_q, device)
    mu_ood = mu_ood.detach().cpu().numpy()
    var_ood = var_ood.detach().cpu().numpy()
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy", mu_ood)
    np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy", var_ood)

    mu_train, var_train = evaluate_laplace(model, inference_model, train_loader, mu_q, sigma_q, device)
    mu_train = mu_train.detach().cpu()
    var_train = var_train.detach().cpu()
    id_labels = torch.cat([batch[1] for batch in id_loader]).cpu()
    train_labels = torch.cat([batch[1] for batch in train_loader]).cpu()

    results = AccuracyCalculator(include=("mean_average_precision", "precision_at_1"), knn_func=CustomKNN(LpDistance()), k=k)\
        .get_accuracy(torch.tensor(mu_id), mu_train, id_labels.squeeze(), train_labels.squeeze(), embeddings_come_from_same_source=False)
    logging.info(f"Post-hoc MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"Post-hoc Accuracy: {100*results['precision_at_1']:.2f}%")

    results = test_model_expected_distance(torch.tensor(mu_id), torch.tensor(var_id), id_labels, mu_train, var_train, train_labels, k=k)
    logging.info(f"Post-hoc MAP@{k} with ED: {results['mean_average_precision']:.2f}")
    logging.info(f"Post-hoc Accuracy with ED: {100*results['precision_at_1']:.2f}%")

    # PLEASE don't run this before you run evaluate_laplace
    samples = sample_nn_weights(mu_q, sigma_q)
    accs = get_sample_accuracy(train_loader.dataset, id_loader.dataset, model, inference_model, samples, device)
    accs = {k: [dic[k] for dic in accs] for k in accs[0]}
    for key, val in accs.items():
        print(f"Post-hoc {key}: {val}")
    vector_to_parameters(mu_q, inference_model.parameters())

    # id_title = id_module.name
    # id_label = id_title.lower()

    # ood_title = ood_module.name
    # ood_label = ood_title.lower()

    # mu_id = np.load(f"results/{method}/{id_label}/id_laplace_mu.npy")
    # var_id = np.load(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy")
    # mu_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy")
    # var_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy")

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

    run_posthoc(128, data.CIFAR10DataModule, data.SVHNDataModule, models.ConvNet)
    run_posthoc(128, data.FashionMNISTDataModule, data.MNISTDataModule, models.FashionMNISTConvNet)

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
