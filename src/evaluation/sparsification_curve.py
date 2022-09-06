"""
Script to plot the sparsification curve for a model on a dataset.

Usage:
    python3 -m src.evaluation.sparsification_curve --model PFE --model_path path/to/model --dataset MNIST --embedding_size XXX --batch_size XXX

Example:
    No debugger:
        python3 -m src.evaluation.sparsification_curve --model PFE --model_path models/PFE_MNIST.pth --dataset MNIST --embedding_size 128


    With debugger:
        python3 -m debugpy --listen 10.66.12.19:1332 src/evaluation/sparsification_curve.py --model PFE --model_path models/PFE/FashionMNIST/contrastive/model.pth --dataset FashionMNIST --embedding_size 6 --loss contrastive
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils import inference
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from src.utils import load_model
from tqdm import tqdm

load_dotenv()

root = Path(__file__).parent.parent.parent.absolute()
data_dir = Path(os.getenv("DATA_DIR"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # Argparse setup
    parser = argparse.ArgumentParser(description="Calibration curve")

    # Add arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["DUL", "HIB", "PFE", "Laplace"],
        help="Model to use",
    )
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MNIST", "CIFAR10", "CASIA", "FashionMNIST"],
        help="Dataset to use",
        default="MNIST",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument(
        "--embedding_size", type=int, help="Embedding size", default=512
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["contrastive", "largemargin"],
        help="Loss to use for PFE",
        default="contrastive",
    )

    return parser.parse_args()


def load(model_name, model_path, dataset, embedding_size, batch_size, loss):
    # Load model
    model_file = root / model_path
    path = model_file.parent

    model = load_model(model_name, dataset, embedding_size, model_file, loss=loss)
    model = model.to(device)
    model.eval()

    # Load dataset
    sampler = None
    if dataset == "MNIST":
        data_module = MNISTDataModule
    elif dataset == "CIFAR10":
        data_module = CIFAR10DataModule
    elif dataset == "CASIA":
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"
    elif dataset == "FashionMNIST":
        data_module = FashionMNISTDataModule

    data_module = data_module(
        data_dir,
        batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        sampler=sampler,
    )

    data_module.setup()

    data_loader = data_module.test_dataloader()

    run(model, data_loader, path, model_name, dataset)


def run(model, data_loader, path, model_name, dataset_name, run_name=""):
    knn_func = inference.CustomKNN(distance=distances.LpDistance())

    metric = AccuracyCalculator(
        include=("precision_at_1",),
        k="max_bin_count",
        device=device,
        knn_func=knn_func,
    )

    accuracies = []

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.to(device)
            target = target.to(device)

            mu, sigma = model(image)

            # Start with all images and remove the highest uncertainty image until
            # there is only 10 images with the lowest uncertainty left
            acc_temp = []

            for i in range(mu.shape[0], 10, -1):
                # Find lowest i element in uncertainty
                _, indices = torch.topk(sigma.sum(dim=1), i, largest=False)

                # Get query elements with lowest uncertainty
                lowest_query = mu[indices]
                lowest_target = target[indices]

                # Compute accuracy where high uncertainty elements are removed
                metrics = metric.get_accuracy(
                    query=lowest_query,
                    reference=lowest_query,
                    query_labels=lowest_target,
                    reference_labels=lowest_target,
                    embeddings_come_from_same_source=True,
                )

                acc_temp.append(metrics["precision_at_1"])

            accuracies.append(torch.tensor(acc_temp))

    # Ignore last element as it may not have the same batch size
    accuracies = torch.stack(accuracies[:-1], dim=0)
    # Calculate average over batches
    accuracies = accuracies.mean(dim=0).cpu().numpy()

    # Calculate AUSC (Area Under the Sparsification Curve)
    ausc = np.trapz(accuracies, dx=1 / len(accuracies))

    # Plot sparsification curve
    fig, ax = plt.subplots()

    # X-axis of % of elements removed
    x = np.arange(0, accuracies.shape[0]) / accuracies.shape[0] * 100

    ax.plot(x, accuracies)

    ax.set(
        xlabel="Filter Out Rate (%)",
        ylabel="Accuracy",
        title=f"Sparsification curve for {model_name} ({run_name}) on {dataset_name}",
    )

    # Add text box with area under the curve
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    textstr = f"AUSC: {ausc:.2f}"
    ax.text(
        0.75,
        0.35,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    # Save figure
    fig.savefig(path / "sparsification_curve.png")

    # Save uncertainty calibration results
    metrics = {
        "ausc": float(ausc),
        "accuracies": accuracies.tolist(),
        "filter_out_rate": x.tolist(),
    }
    with open(path / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f)

    return float(ausc)


if __name__ == "__main__":
    args = parse_args()

    load(
        args.model,
        args.model_path,
        args.dataset,
        args.embedding_size,
        args.batch_size,
        args.loss,
    )
