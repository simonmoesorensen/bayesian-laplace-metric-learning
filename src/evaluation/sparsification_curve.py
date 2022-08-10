"""
Script to plot the sparsification curve for a model on a dataset.

Usage:
    python3 -m src.evaluation.sparsification_curve --model PFE --model_path path/to/model --dataset MNIST --embedding_size XXX --batch_size XXX

Example:
    No debugger:
        python3 -m src.evaluation.sparsification_curve --model PFE --model_path models/PFE_MNIST.pth --dataset MNIST --embedding_size 128


    With debugger:
        python3 -m debugpy --listen 10.66.12.19:1332 src/evaluation/sparsification_curve.py --model PFE --model_path models/PFE_MNIST.pth --dataset MNIST --embedding_size 128
"""

import argparse

from src.utils import load_model
from pathlib import Path
from src.data_modules import MNISTDataModule
from src.data_modules import CIFAR10DataModule
from src.data_modules import CasiaDataModule
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
from pytorch_metric_learning.utils import inference
from pytorch_metric_learning import distances
import numpy as np
import matplotlib.pyplot as plt
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

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

    return parser.parse_args()


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

    # Plot sparsification curve
    fig, ax = plt.subplots()

    # X-axis of % of elements removed
    x = np.arange(0, accuracies.shape[0]) / accuracies.shape[0] * 100

    ax.plot(x, accuracies)

    ax.set(
        xlabel="Filter Out Rate (%)",
        ylabel="Accuracy",
        title=f"Sparsification curve for {args.model} on {args.dataset}",
    )

    # Save dir
    save_dir = root / "outputs" / args.model / "figures" / args.dataset / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir / "sparsification_curve.png")


if __name__ == "__main__":
    args = parse_args()
    run(args)
