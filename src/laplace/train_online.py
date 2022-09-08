import json
import logging
from pathlib import Path

import pandas as pd
import torch
from pytorch_metric_learning.utils.inference import CustomKNN
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet

from src.laplace.config import parse_args
from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
)
from src.recall_at_k import AccuracyRecall
from src.distances import ExpectedSquareL2Distance
from pytorch_metric_learning.distances import LpDistance
import logging
import numpy as np
import matplotlib.pyplot as plt

from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch
from tqdm import tqdm
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.laplace.hessian.layerwise import ContrastiveHessianCalculator
from src.visualize import calibration_curves, visualize_all


def sample_neural_network_wights(parameters, posterior_scale, n_samples=32):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_online(
    epochs, freq, nn_samples, device, lr, train_loader, margin, net, net_inference
):
    contrastive_loss = losses.ContrastiveLoss(neg_margin=margin)
    # miner = miners.MultiSimilarityMiner()
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.ALL,
        neg_strategy=miners.BatchEasyHardMiner.ALL,
    )
    hessian_calculator = ContrastiveHessianCalculator(margin=margin, device=device)

    hessian_calculator.init_model(net_inference)

    num_params = sum(p.numel() for p in net_inference.parameters())
    dataset_size = len(train_loader.dataset)
    num_batches = len(train_loader)
    alpha = lr * num_batches
    scale = 1.0
    prior_prec = 1.0

    optim = Adam(net.parameters(), lr=lr)

    # H = 1e10 * torch.ones((num_params,), device=device)
    H = torch.ones((num_params,), device=device)
    posterior_precision = H * scale + prior_prec
    sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)

    for epoch in range(epochs):
        epoch_losses = []
        compute_hessian = epoch % freq == 0

        if compute_hessian:
            hs = 0

        for i, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            mu_q = parameters_to_vector(net_inference.parameters())

            sampled_nn = sample_neural_network_wights(
                mu_q, sigma_q, n_samples=nn_samples
            )

            con_loss = 0
            if compute_hessian:
                h = 0

            for nn_i in sampled_nn:
                vector_to_parameters(nn_i, net_inference.parameters())
                output = net(x)
                hard_pairs = miner(output, y)

                if compute_hessian:
                    # Adjust hessian to the batch size
                    scaler = (
                        dataset_size**2
                        / (len(hard_pairs[0]) + len(hard_pairs[2])) ** 2
                    )
                    hessian_sample = hessian_calculator.compute_batch_pairs(hard_pairs)
                    hs += hessian_sample * scaler / nn_samples

                con_loss += contrastive_loss(output, y, hard_pairs)
            # if compute_hessian:
            #     h /= nn_samples
            #     hs += h


            vector_to_parameters(mu_q, net_inference.parameters())
            if compute_hessian and nn_samples == 1:
                output = net(x)
                hard_pairs = miner(output, y)
                scaler = (
                    dataset_size**2
                    / (len(hard_pairs[0]) + len(hard_pairs[2])) ** 2
                )
                hessian_batch = hessian_calculator.compute_batch_pairs(hard_pairs)
                
                hs += hessian_batch * scaler

            # if compute_hessian:
                # H = hs / (i + 1)
                # posterior_precision = H * scale + prior_prec
                # sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)

            con_loss /= nn_samples
            loss = con_loss
            vector_to_parameters(mu_q, net_inference.parameters())
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
    
        if compute_hessian:
            hs = torch.clamp(hs, min=0)
            # H = hs / num_batches
            H = (1 - alpha) * H + hs / num_batches
            posterior_precision = H * scale + prior_prec
            sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)


        loss_mean = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss_mean=} for {epoch=}")

    mu_q = parameters_to_vector(net_inference.parameters())
    posterior_precision = H * scale + prior_prec
    sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)
    return mu_q, sigma_q


def evaluate_laplace(net, inference_net, loader, mu_q, sigma_q, device="cpu"):
    samples = sample_nn_weights(mu_q, sigma_q)

    pred_mean, pred_var = generate_predictions_from_samples_rolling(loader, samples, net, inference_net, device)
    return pred_mean, pred_var



def get_single_sample_pred(full_model, loader, device) -> torch.Tensor:
    preds = []
    for x, _ in iter(loader):
        with torch.inference_mode():
            pred = full_model(x.to(device))
        preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds

def generate_predictions_from_samples_rolling(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    """
    Welford's online algorithm for calculating mean and variance.
    """
    if inference_model is None:
        inference_model = full_model

    N = len(weight_samples)

    vector_to_parameters(weight_samples[0, :], inference_model.parameters())
    mean = get_single_sample_pred(full_model, loader, device)
    msq = 0.0
    delta = 0.0

    for i, net_sample in enumerate(weight_samples[1:, :]):
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = get_single_sample_pred(full_model, loader, device)
        delta = sample_preds - mean
        mean += delta / (i + 1)
        msq += delta * delta

    variance = msq / (N - 1)
    return mean, variance


def sample_nn_weights(parameters, posterior_scale, n_samples=16):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def visualize(
    id_mu,
    id_sigma,
    id_images,
    id_targets,
    ood_mu,
    ood_sigma,
    ood_images,
    dataset,
    vis_path,
):
    print("=" * 60, flush=True)
    print("Visualizing...")

    latent_dim = id_mu.shape[1]
    # Visualize

    visualize_all(
        id_mu,
        id_sigma.sqrt(),
        id_images,
        ood_mu,
        ood_sigma.sqrt(),
        ood_images,
        vis_path,
        prefix=f"{latent_dim}",
    )

    print("Running calibration curve")
    run_calibration_curve(
        id_targets, id_mu, id_sigma, 50, vis_path, "post_hoc", dataset
    )

    print("Running sparsification curve")
    run_sparsification_curve(id_targets, id_mu, id_sigma, vis_path, "post_hoc", dataset)

    # # Save metrics
    # with open(vis_path / "metrics.json", "w") as f:
    #     json.dump(self.metrics.get_dict(), f)


def run_sparsification_curve(targets, mus, sigmas, path, model_name, dataset_name):
    knn_func = CustomKNN(distance=distances.LpDistance())

    metric = AccuracyCalculator(
        include=("precision_at_1",),
        k="max_bin_count",
        device=device,
        knn_func=knn_func,
    )
    latent_dim = mus.shape[-1]

    accuracies = []

    for target, mu, sigma in DataLoader(TensorDataset(targets, mus, sigmas), 128):

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
        title=f"Sparsification curve for {model_name} on {dataset_name}",
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
    pd.DataFrame.from_dict({"ausc": metrics["ausc"]}, orient="index").assign(
        dim=latent_dim
    ).to_csv(path / f"metrics.csv", mode="a", header=False)


def run_calibration_curve(
    targets, mus, sigmas, samples, path, model_name, dataset_name
):
    knn_func = CustomKNN(distance=distances.LpDistance())
    latent_dim = mus.shape[-1]

    predicted = []
    confidences = []
    for target, mu, sigma in DataLoader(TensorDataset(targets, mus, sigmas), 128):
        cov = torch.diag_embed(sigma)
        pdist = torch.distributions.MultivariateNormal(mu, cov)

        pred_labels = []

        for _ in range(samples):
            # Save space by sampling once every iteration instead of all in one go
            sample = pdist.sample()

            # knn_func(query, k, reference, shares_datapoints)
            _, indices = knn_func(sample, 1, sample, True)
            pred_labels.append(target[indices].squeeze())

        pred_labels = torch.stack(pred_labels, dim=1)
        pred = torch.mode(pred_labels, dim=1).values

        predicted.append(pred)
        confidences.append((pred == pred_labels.T).to(torch.float16).mean(dim=0))

    predicted = torch.cat(predicted, dim=0)
    confidences = torch.cat(confidences, dim=0)

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
        title=f"ECE curve for {model_name} on {dataset_name}",
    )

    # Add grid
    ax.grid(True, linestyle="dotted")

    # Save dir
    fig.savefig(path / "calibration_curve.png")

    # Save metrics
    metrics = {
        "ece": float(ece),
        "acc": acc.tolist(),
        "conf": conf.tolist(),
    }

    with open(path / "calibration_curve.json", "w") as f:
        json.dump(metrics, f)
    pd.DataFrame.from_dict({"ece": metrics["ece"]}, orient="index").assign(
        dim=latent_dim
    ).to_csv(path / f"metrics.csv", mode="a", header=False)

def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    sampler = None

    if args.dataset == "CIFAR10":
        model = CIFAR10ConvNet(args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "FashionMNIST":
        model = FashionMNISTConvNet(args.embedding_size)
        data_module = FashionMNISTDataModule
    else:
        raise ValueError("Dataset not supported")

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        sampler=sampler,
    )
    model.to(device)
    data_module.setup()

    name = f"{args.hessian}_{args.embedding_size}_seed_{args.random_seed}"

    pretrained_path = Path(f"outputs/Online/checkpoints") / args.dataset / name
    figure_path = Path(f"outputs/Online/figures") / args.dataset / name
    print(f"{pretrained_path=}")
    print(f"{figure_path=}")
    pretrained_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    inference_model = getattr(model, args.inference_model)

    mu_q, sigma_q = train_online(
        args.num_epoch,
        args.hessian_freq,
        1,
        device,
        args.lr,
        data_module.train_dataloader(),
        args.margin,
        model,
        inference_model,
    )
    torch.save(
        model.state_dict(), f"{pretrained_path}/state_dict_{args.embedding_size}.pt"
    )
    torch.save(mu_q, f"{pretrained_path}/mu_q_{args.embedding_size}.pt")
    torch.save(sigma_q, f"{pretrained_path}/sigma_q_{args.embedding_size}.pt")
    # model.load_state_dict(torch.load(f"{pretrained_path}/state_dict_{args.embedding_size}.pt", map_location=device))
    # mu_q = torch.load(f"{pretrained_path}/mu_q_{args.embedding_size}.pt", map_location=device)
    # sigma_q = torch.load(f"{pretrained_path}/sigma_q_{args.embedding_size}.pt", map_location=device)

    mu_id, var_id = evaluate_laplace(
        model, inference_model, data_module.test_dataloader(), mu_q, sigma_q, device
    )
    mu_id = mu_id.detach().cpu()  # .numpy()
    var_id = var_id.detach().cpu()  # .numpy()
    id_images = (
        torch.cat([x for x, _ in data_module.test_dataloader()], dim=0).detach().cpu()
    )  # .numpy()
    id_labels = (
        torch.cat([y for _, y in data_module.test_dataloader()], dim=0).detach().cpu()
    )  # .numpy()

    mu_ood, var_ood = evaluate_laplace(
        model, inference_model, data_module.ood_dataloader(), mu_q, sigma_q, device
    )
    mu_ood = mu_ood.detach().cpu()  # .numpy()
    var_ood = var_ood.detach().cpu()  # .numpy()
    ood_images = (
        torch.cat([x for x, _ in data_module.ood_dataloader()], dim=0).detach().cpu()
    )  # .numpy()

    mu_train, var_train = evaluate_laplace(
        model, inference_model, data_module.train_dataloader(), mu_q, sigma_q, device
    )
    mu_train = mu_train.detach().cpu()  # .numpy()
    var_train = var_train.detach().cpu()  # .numpy()
    train_labels = (
        torch.cat([y for _, y in data_module.train_dataloader()], dim=0).detach().cpu()
    )  # .numpy()
    results = AccuracyRecall(
        include=("mean_average_precision", "precision_at_1", "recall_at_k"),
        k=5,
        device=device,
        knn_func=CustomKNN(LpDistance()),
    ).get_accuracy(
        mu_id,
        mu_train,
        id_labels.squeeze(),
        train_labels.squeeze(),
        embeddings_come_from_same_source=False,
    )
    pd.DataFrame.from_dict(results, orient="index").assign(
        dim=args.embedding_size
    ).to_csv(f"{figure_path}/metrics.csv", mode="a", header=False)
    results = AccuracyRecall(
        include=("mean_average_precision", "precision_at_1", "recall_at_k"),
        k=5,
        device=device,
        knn_func=CustomKNN(ExpectedSquareL2Distance()),
    ).get_accuracy(
        torch.stack((mu_id, var_id), dim=-1),
        torch.stack((mu_train, var_train), dim=-1),
        id_labels.squeeze(),
        train_labels.squeeze(),
        embeddings_come_from_same_source=False,
    )
    pd.DataFrame.from_dict(
        {
            "mean_average_precision_expected": results["mean_average_precision"],
            "precision_at_1_expected": results["precision_at_1"],
            "recall_at_k_expected": results["recall_at_k"],
        },
        orient="index",
    ).assign(dim=args.embedding_size).to_csv(
        f"{figure_path}/metrics.csv", mode="a", header=False
    )

    visualize(
        mu_id,
        var_id,
        id_images,
        id_labels,
        mu_ood,
        var_ood,
        ood_images,
        args.dataset,
        Path(figure_path),
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(parse_args())
