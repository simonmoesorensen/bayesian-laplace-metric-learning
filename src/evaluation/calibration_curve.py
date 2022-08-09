"""
Script to plot the calibration curve for the model

Usage:
    python3 -m src.evaluation.calibration_curve --model PFE --model_path path/to/model --dataset MNIST --embedding_size XXX --batch_size XXX

Example:
    No debugger:
        python3 -m src.evaluation.calibration_curve --model PFE --model_path models/PFE_MNIST.pth --dataset MNIST --embedding_size 128


    With debugger:
        python3 -m debugpy --listen 10.66.20.1:1332 src/evaluation/calibration_curve.py --model PFE --model_path models/PFE_MNIST.pth --dataset MNIST --embedding_size 128
"""

import argparse
from tkinter import Grid

from src.utils import load_model
from pathlib import Path
from src.data_modules import MNISTDataModule
from src.data_modules import CIFAR10DataModule
from src.data_modules import CasiaDataModule
from dotenv import load_dotenv
import os
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm
from pytorch_metric_learning.utils import inference
from pytorch_metric_learning import distances
import numpy as np
import matplotlib.pyplot as plt

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
        choices=["MNIST", "CIFAR10", "CASIA"],
        help="Dataset to use",
        default="MNIST",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument(
        "--embedding_size", type=int, help="Embedding size", default=512
    )

    parser.add_argument("--samples", type=int, help="Number of samples", default=30)
    return parser.parse_args()


def calibration_curves(targets, confidences, preds, bins=10, fill_nans=False):
    targets = targets.cpu().numpy()
    confidences = confidences.cpu().numpy()
    preds = preds.cpu().numpy()

    real_probs = np.zeros((bins,))
    pred_probs = np.zeros((bins,))
    bin_sizes = np.zeros((bins,))

    _, lims = np.histogram(confidences, range=(0.0, 1.0), bins=bins)
    for i in range(bins):
        lower, upper = lims[i], lims[i + 1]
        mask = (lower <= confidences) & (confidences < upper)

        targets_in_range = targets[mask]
        preds_in_range = preds[mask]
        probs_in_range = confidences[mask]
        n_in_range = preds_in_range.shape[0]

        range_acc = (
            np.sum(targets_in_range == preds_in_range) / n_in_range
            if n_in_range > 0
            else 0
        )
        range_prob = np.sum(probs_in_range) / n_in_range if n_in_range > 0 else 0

        real_probs[i] = range_acc
        pred_probs[i] = range_prob
        bin_sizes[i] = n_in_range

    bin_weights = bin_sizes / np.sum(bin_sizes)
    ece = np.sum(np.abs(real_probs - pred_probs) * bin_weights)

    if fill_nans:
        return ece, real_probs, pred_probs, bin_sizes
    return ece, real_probs[bin_sizes > 0], pred_probs[bin_sizes > 0], bin_sizes


def run(args):
    # Load model
    path = root / args.model_path
    model = load_model(args.model, args.dataset, args.embedding_size, path)
    model = model.to(device)
    model.eval()

    # Load dataset
    sampler = None
    if args.dataset == "MNIST":
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        data_module = CIFAR10DataModule
    elif args.dataset == "CASIA":
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"

    data_module = data_module(
        data_dir,
        args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        sampler=sampler,
    )

    data_module.setup()

    data_loader = data_module.test_dataloader()

    knn_func = inference.CustomKNN(distance=distances.LpDistance())

    predicted = []
    confidences = []
    targets = []

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.to(device)
            target = target.to(device)

            mu, sigma = model(image)

            cov = torch.diag_embed(sigma)
            pdist = torch.distributions.MultivariateNormal(mu, cov)

            pred_labels = []

            for _ in range(args.samples):
                # Save space by sampling once every iteration instead of all in one go
                sample = pdist.sample()

                # knn_func(query, k, reference, shares_datapoints)
                _, indices = knn_func(sample, 1, sample, True)
                pred_labels.append(target[indices].squeeze())

            pred_labels = torch.stack(pred_labels, dim=1)
            pred = torch.mode(pred_labels, dim=1).values

            predicted.append(pred)
            confidences.append((pred == pred_labels.T).to(torch.float16).mean(dim=0))
            targets.append(target)

    predicted = torch.cat(predicted, dim=0)
    confidences = torch.cat(confidences, dim=0)
    targets = torch.cat(targets, dim=0)

    # Plotting ECE
    bins = 10

    ece, acc, conf, bin_sizes = calibration_curves(
        targets=targets,
        confidences=confidences,
        preds=predicted,
        bins=bins,
        fill_nans=True,
    )

    fig, ax = plt.subplots()

    # Plot ECE
    ax.plot(conf, acc, label="ECE")

    # Add histogram of confidences scaled between 0 and 1
    confidences = confidences.cpu().numpy()
    ax.hist(
        confidences,
        bins=bins,
        density=True,
        label="Distribution of confidences",
        alpha=0.5,
    )

    # Plot identity line
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "k--", label="Best fit")

    # Add textbox with ECE value
    textstr = f"ECE: {ece:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.75,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # Add legend
    ax.legend()

    # Set axis options
    ax.set(
        xlim=[0, 1],
        ylim=[0, 1],
        xlabel="Confidence",
        ylabel="Accuracy",
        title=f"ECE curve for {args.model} on {args.dataset}",
    )

    # Add grid
    ax.grid(True, linestyle="dotted")

    # Save dir
    save_dir = root / "outputs" / args.model / "figures" / args.dataset / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir / "calibration_curve.png")


if __name__ == "__main__":
    args = parse_args()
    run(args)
