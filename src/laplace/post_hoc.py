import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.laplace.models import L2Norm
from src.baselines.Backbone.models import (
    Casia_Backbone,
    CIFAR10_Backbone,
    MNIST_Backbone,
)
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from src.distances import ExpectedSquareL2Distance
from src.evaluation.calibration_curve import calibration_curves
from src.laplace.hessian.layerwise import (
    ContrastiveHessianCalculator,
    FixedContrastiveHessianCalculator,
)
from src.laplace.config import parse_args

# from src.laplace.post_hoc import post_hoc
from src.miners import AllCombinationsMiner, AllPositiveMiner
from src.recall_at_k import AccuracyRecall
from src.visualize import visualize_all

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]
    torch.manual_seed(args.random_seed)

    sampler = None

    if args.dataset == "MNIST":
        model = MNIST_Backbone(embedding_size=args.embedding_size)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10_Backbone(embedding_size=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        model = Casia_Backbone(embedding_size=args.embedding_size)
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"
    elif args.dataset == "FashionMNIST":
        model = MNIST_Backbone(embedding_size=args.embedding_size)
        data_module = FashionMNISTDataModule
    else:
        raise ValueError("Dataset not supported")

    model = model[0]
    model.linear.add_module("l2norm", L2Norm())
    state_dict = torch.load(args.model_path)

    new_state_dict = {}
    for key in state_dict:
        if key.startswith("0."):
            new_state_dict[key[2:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    vis_path = Path("outputs") / "PostHoc" / "figures" / args.dataset / args.hessian
    vis_path.mkdir(parents=True, exist_ok=True)

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

    inference_model = getattr(model, args.inference_model)

    mu_q, sigma_q = post_hoc(
        model,
        inference_model,
        data_module.train_dataloader(),
        margin=args.margin,
        device=device,
        method=args.hessian,
    )
    torch.save(
        sigma_q, f"sigma_q_{args.dataset}_{args.embedding_size}_{args.hessian}.pt"
    )

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
    ).to_csv(vis_path / "metrics.csv", mode="a", header=False)
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
        vis_path / "metrics.csv", mode="a", header=False
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
        vis_path,
    )

    # mu_id = mu_id.detach().cpu().numpy()
    # var_id = var_id.detach().cpu().numpy()
    # mu_ood = mu_ood.detach().cpu().numpy()
    # var_ood = var_ood.detach().cpu().numpy()

    # fig, ax = plot_ood(mu_id, var_id, mu_ood, var_ood)
    # fig.tight_layout()
    # fig.savefig(f"ood_plot.png")

    # metrics = compute_and_plot_roc_curves(".", var_id, var_ood)
    # print(metrics)


def log_det_ratio(hessian, prior_prec):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def scatter(mu_q, prior_precision_diag):
    return (mu_q * prior_precision_diag) @ mu_q


def log_marginal_likelihood(mu_q, hessian, prior_prec):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = -0.5 * (
        log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec)
    )
    return neg_log_marglik


def optimize_prior_precision(mu_q, hessian, prior_prec, n_steps=100):

    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()

    return prior_prec


def post_hoc(
    model: nn.Module,
    inference_model: nn.Module,
    train_loader: DataLoader,
    margin: float,
    device="cpu",
    method="full",
):
    """
    Run post-hoc laplace on a pretrained metric learning model.
    """
    if method == "positives":
        calculator = ContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllPositiveMiner()
    elif method == "fixed":
        calculator = FixedContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllCombinationsMiner()
    elif method == "full":
        calculator = ContrastiveHessianCalculator(margin=margin, device=device)
        miner = AllCombinationsMiner()
    else:
        raise ValueError(f"Unknown method: {method}")

    dataset_size = len(train_loader.dataset)

    calculator.init_model(inference_model)
    h = 0
    with torch.no_grad():
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            hard_pairs = miner(output, y)

            # Total number of possible pairs / number of pairs in our batch
            scaler = dataset_size**2 / x.shape[0] ** 2
            hessian = calculator.compute_batch_pairs(hard_pairs)
            h += hessian * scaler

    if (h < 0).sum():
        logging.warning("Found negative values in Hessian.")

    # Scale by number of batches
    h /= len(train_loader)

    h = torch.maximum(h, torch.tensor(0))

    logging.info(
        f"{100 * calculator.zeros / calculator.total_pairs:.2f}% of pairs are zero."
    )
    logging.info(
        f"{100 * calculator.negatives / calculator.total_pairs:.2f}% of pairs are negative."
    )

    map_solution = parameters_to_vector(inference_model.parameters())

    scale = 1.0
    prior_prec = 1.0
    prior_prec = optimize_prior_precision(map_solution, h, torch.tensor(prior_prec))
    posterior_precision = h * scale + prior_prec
    posterior_scale = 1.0 / (posterior_precision.sqrt() + 1e-6)

    return map_solution, posterior_scale


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
        [id_mu],
        [id_sigma.sqrt()],
        [id_images],
        [ood_mu],
        [ood_sigma.sqrt()],
        [ood_images],
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


from typing import Tuple

import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from torch.nn.utils.convert_parameters import vector_to_parameters


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    for i, net_sample in tqdm(enumerate(weight_samples[1:, :])):
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = get_single_sample_pred(full_model, loader, device)
        delta = sample_preds - mean
        mean += delta / (i + 1)
        msq += delta * delta

    variance = msq / (N - 1)
    return mean, variance


def sample_nn_weights(parameters, posterior_scale, n_samples=100):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def evaluate_laplace(net, inference_net, loader, mu_q, sigma_q, device="cpu"):
    samples = sample_nn_weights(mu_q, sigma_q)

    pred_mean, pred_var = generate_predictions_from_samples_rolling(
        loader, samples, net, inference_net, device
    )
    return pred_mean, pred_var


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchmetrics
from matplotlib.patches import Ellipse
from scipy.stats import hmean

sns.set_theme(style="ticks")
c_id = "b"
c_ood = "r"


def plot_samples(
    mu, sigma_sq, latent1=0, latent2=1, limit=100, ax=None, color="b", label=None
):
    if ax is None:
        _, ax = plt.subplots()

    # np.random.seed(0)
    # indices = np.random.choice(np.arange(mu.shape[0]), size=limit, )
    indices = np.arange(limit)

    ax.scatter(
        mu[indices, latent1],
        mu[indices, latent2],
        s=0.5,
        c=color,
        label=label,
    )
    for index in indices:
        elp = Ellipse(
            (mu[index, latent1], mu[index, latent2]),
            sigma_sq[index, latent1],
            sigma_sq[index, latent2],
            fc="None",
            edgecolor=color,
            lw=0.5,
        )
        ax.add_patch(elp)
    ax.set(
        xlabel=f"Latent dim {latent1}",
        ylabel=f"Latent dim {latent2}",
    )


def plot_histogram(sigma_sq, mean="arithmetic", ax=None, color="b", label=None):
    if ax is None:
        _, ax = plt.subplots()

    if mean == "harmonic":
        mean_sigma_sq = hmean(sigma_sq, axis=1)
    elif mean == "arithmetic":
        mean_sigma_sq = np.mean(sigma_sq, axis=1)
    else:
        raise NotImplementedError

    print(
        f"mean={mean_sigma_sq.mean():.5f}, "
        f"std={mean_sigma_sq.std():.5f}, "
        f"min={mean_sigma_sq.min():.5f}, "
        f"max={mean_sigma_sq.max():.5f}"
    )

    sns.kdeplot(mean_sigma_sq, ax=ax, color=color, label=label)
    ax.set(xlabel="Variance")


def plot_ood(mu_id, var_id, mu_ood, var_ood):
    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))
    plot_samples(mu_id, var_id, limit=100, color=c_id, label="ID", ax=ax[0])
    plot_histogram(var_id, color=c_id, label="ID", ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color=c_ood, label="OOD", ax=ax[0])
    plot_histogram(var_ood, color=c_ood, label="OOD", ax=ax[1])
    ax[1].get_yaxis().set_ticks([])
    ax[1].set_ylabel(None)
    ax[1].legend()
    return fig, ax


def compute_and_plot_roc_curves(path, id_sigma, ood_sigma, pre_fix=""):

    id_sigma = np.reshape(id_sigma, (id_sigma.shape[0], -1))
    ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

    id_sigma, ood_sigma = id_sigma.sum(axis=1), ood_sigma.sum(axis=1)

    pred = np.concatenate([id_sigma, ood_sigma])
    target = np.concatenate([[0] * len(id_sigma), [1] * len(ood_sigma)])

    # plot roc curve
    roc = torchmetrics.ROC(num_classes=1)
    fpr, tpr, thresholds = roc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    fig.tight_layout()
    fig.savefig(f"{path}/{pre_fix}ood_roc_curve.png")

    # save data
    # data = pd.DataFrame(
    #     np.concatenate([pred[:, None], target[:, None]], axis=1),
    #     columns=["sigma", "labels"],
    # )
    # data.to_csv(f"figures/{path}/{pre_fix}ood_roc_curve_data.csv")

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(recall, precision)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
    )
    fig.tight_layout()
    fig.savefig(f"{path}/{pre_fix}ood_precision_recall_curve.png")

    metrics = {}

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)
    metrics["auprc"] = float(auprc_score.numpy())

    # compute false positive rate at 80
    # num_id = len(id_sigma)
    # for p in range(0, 100, 10):
    #     # if there is no difference in variance
    #     try:
    #         metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
    #     except:
    #         metrics[f"fpr{p}"] = "none"
    #     else:
    #         continue

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )
    metrics["auroc"] = float(auroc_score.numpy())

    return metrics


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(parse_args())
