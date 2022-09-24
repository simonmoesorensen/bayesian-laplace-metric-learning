import json
from turtle import up

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from matplotlib.patches import Ellipse
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torch.utils.data import DataLoader, TensorDataset
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning import distances
from tqdm import tqdm
import time
from pathlib import Path, PosixPath
from sklearn.neighbors import NearestNeighbors

sns.set()

c_id = "b"
c_ood = "r"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vis_path(dict_log):
    
    # Set path
    vis_path = Path(dict_log["path"]) / dict_log["dataset"] / dict_log["name"] / f"epoch_{dict_log['epoch']}"

    vis_path.mkdir(parents=True, exist_ok=True)
    
    return vis_path

def visualize(dict_, dict_ood, dict_other, dict_log, prefix):

    print("=" * 60, flush=True)
    print("Visualizing...")
    metrics = {}
    
    vis_path = get_vis_path(dict_log)

    if "hessian" in dict_other:
        plt.plot(dict_other["hessian"].cpu().numpy())
        plt.yscale("log")
        plt.savefig(vis_path / "hessian.png");
        plt.close(); plt.cla(); plt.clf();

    prob_model = dict_["z_sigma"] is not None and len(dict_["z_sigma"]) > 0
    if prob_model:
        
        if dict_ood is not None:
        
            # Visualize
            tmp = ood_visualisations(
                dict_["z_mu"], 
                dict_["z_sigma"],
                dict_["images"],
                dict_ood["z_mu"], 
                dict_ood["z_sigma"],
                dict_ood["images"],
                vis_path, prefix
            )
            for k in tmp:
                metrics[k] = tmp[k]

        model_name, dataset_name, run_name = get_names(vis_path)

        # calibration curve
        print("Running calibration curve")
        ece = plot_calibration_curve(
            dict_["labels"],
            dict_["z_samples"],
            vis_path,
            model_name,
            dataset_name,
            run_name,
        )
        metrics[f"{prefix}/ece"] = ece

        # Sparsification curve
        print("Running sparsification curve")
        ausc = plot_sparsification_curve(
            dict_["labels"], 
            dict_["z_mu"], 
            dict_["z_sigma"], 
            vis_path, 
            model_name, 
            dataset_name, 
            run_name
        )
        metrics[f"{prefix}/ausc"] = ausc

    return metrics


def visualize_top_5(id_sigma, id_images, ood_sigma, ood_images, vis_path, prefix, n=5):
    """Visualize the top 5 highest and lowest variance images"""
    id_sigma = id_sigma.numpy()
    id_images = id_images.numpy()
    ood_sigma = ood_sigma.numpy()
    ood_images = ood_images.numpy()

    model_name, dataset, run_name = get_names(vis_path)

    # Get colormap
    channels = id_images.shape[1]

    if channels == 1:
        cmap = "gray"
    elif channels == 3:
        cmap = None

    # Get l2 norm of ID variances
    id_var_mu = np.mean(id_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of ID variances
    top_5_id = (-id_var_mu).argsort()[:n]
    bot_5_id = (-id_var_mu).argsort()[-n:]

    # Get l2 norm of OOD variances
    ood_var_mu = np.mean(ood_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of OOD variances
    top_5_ood = (-ood_var_mu).argsort()[:n]
    bot_5_ood = (-ood_var_mu).argsort()[-n:]

    # plot top and bottom 5 images
    rows = 4
    columns = n
    fig = plt.figure(figsize=(10, 7))
    counter = 0
    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = id_images[top_5_id[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"ID V={id_var_mu[top_5_id[col]]:.2E}")
        if col == 0:
            plt.ylabel("Top 5 var ID")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = id_images[bot_5_id[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"ID V={id_var_mu[bot_5_id[col]]:.2E}")
        if col == 0:
            plt.ylabel("Bot 5 var ID")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = ood_images[top_5_ood[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"OOD V={ood_var_mu[top_5_ood[col]]:.2E}")
        if col == 0:
            plt.ylabel("Top 5 var OOD")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = ood_images[bot_5_ood[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"OOD V={ood_var_mu[bot_5_ood[col]]:.2E}")
        if col == 0:
            plt.ylabel("Bot 5 var OOD")
        counter += 1

    plt.suptitle(
        f"Top and bottom 5 variance images for model {model_name} ({run_name}) on dataset {dataset}"
    )

    fig.savefig(vis_path / f"{prefix}top_bot_5_var.png")


def ood_visualisations(
    id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
):

    if not prefix.endswith("_"):
        prefix += "_"

    # visualize top 5 and bottom 5 variance images
    visualize_top_5(id_sigma, id_images, ood_sigma, ood_images, vis_path, prefix)

    # Visualize
    metrics = plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix)


    plot_ood(id_mu, id_sigma, ood_mu, ood_sigma, vis_path, prefix)
    
    return metrics


def plot_samples(
    mu, sigma_sq, latent1=0, latent2=1, limit=100, ax=None, color="b", label=None
):
    if ax is None:
        _, ax = plt.subplots()

    indices = np.random.choice(np.arange(mu.shape[0]), size=limit)

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


def plot_histogram(sigma_sq, ax=None, color="b", label=None):
    if ax is None:
        _, ax = plt.subplots()

    mean_sigma_sq = np.mean(sigma_sq.numpy(), axis=1)

    sns.kdeplot(mean_sigma_sq, ax=ax, color=color, label=label)
    ax.set(xlabel="Variance")


def plot_ood(mu_id, var_id, mu_ood, var_ood, vis_path, prefix):
    model_name, dataset, run_name = get_names(vis_path)
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_samples(mu_id, var_id, limit=100, color=c_id, label="ID", ax=ax[0])
    plot_histogram(var_id, color=c_id, ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color=c_ood, label="OOD", ax=ax[0])
    plot_histogram(var_ood, color=c_ood, ax=ax[1])
    ax[0].legend()
    fig.suptitle(
        f"ID vs OOD comparison for model {model_name} ({run_name}) on dataset {dataset}"
    )
    fig.savefig(vis_path / f"{prefix}ood_comparison.png")
    return fig, ax


def plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix):
    model_name, dataset, run_name = get_names(vis_path)
    latent_dim = id_sigma.shape[-1]

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

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"OOD ROC Curve for model {model_name} ({run_name}) on dataset {dataset}")
    plt.legend()
    fig.savefig(vis_path / f"{prefix}ood_roc_curve.png")
    plt.close(); plt.cla(); plt.clf();

    # save data
    data = pd.DataFrame(
        np.concatenate([pred[:, None], target[:, None]], axis=1),
        columns=["sigma", "labels"],
    )
    data.to_csv(vis_path / f"{prefix}ood_roc_curve_data.csv")

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"OOD Precision-Recall Curve for model {model_name} ({run_name}) on dataset {dataset}"
    )
    plt.legend()
    fig.savefig(vis_path / f"{prefix}ood_precision_recall_curve.png")
    plt.close(); plt.cla(); plt.clf();

    metrics = {}

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)
    metrics["auprc"] = float(auprc_score.numpy())

    # compute false positive rate at 80
    num_id = len(id_sigma)

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )
    metrics["auroc"] = float(auroc_score.numpy())

    for p in range(0, 100, 10):
        # if there is no difference in variance
        try:
            metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
        except Exception:
            metrics[f"fpr{p}"] = "none"
        else:
            continue

    # save metrics
    with open(vis_path / "ood_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)

    return {"auroc": float(auroc_score.numpy()), "auprc": float(auprc_score.numpy())}


def get_names(vis_path):
    # returns (model name, dataset trained on)
    return (vis_path.parts[-5], vis_path.parts[-3], vis_path.parts[-2])


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
        #range_prob = (upper + lower) / 2

        real_probs[i] = range_acc
        pred_probs[i] = range_prob
        bin_sizes[i] = n_in_range

    bin_weights = bin_sizes / np.sum(bin_sizes)
    ece = np.sum(np.abs(real_probs - pred_probs) * bin_weights)
    
    return ece, real_probs, pred_probs, bin_sizes
    #return ece, real_probs[bin_sizes > 0], pred_probs[bin_sizes > 0], bin_sizes


def plot_calibration_curve(
    targets, samples, path, model_name, dataset_name, run_name
):
    
    N, n_samples, d = samples.shape
    
    pred_labels = []
    print(f"==> Computing ece predictions for {n_samples} samples")
    for i in tqdm(range(n_samples)):
        
        sample_i = samples[:, i, :]
        
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(sample_i)
        dist, idx = neigh.kneighbors(sample_i)
        
        # remove itself from
        dist = dist[:, 1]
        idx = idx[:, 1]
        
        pred_labels.append(targets[idx])
    
    pred_labels = torch.stack(pred_labels)
    predicted, _ = torch.mode(pred_labels, dim=0)   
    confidences = torch.mean((pred_labels == predicted).float(), dim=0)   

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
    ax.plot(np.linspace(0.05,0.95,10).tolist(), acc.tolist(), '-o', label="Calibration curve")

    # Add histogram of confidences scaled between 0 and 1
    confidences = confidences.cpu().numpy()
    ax.hist(
        confidences,
        bins=bins,
        range=(0.0, 1.0),
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
        xlim=[-0.1, 1.1],
        ylim=[-0.1, 1.1],
        xlabel="Confidence",
        ylabel="Accuracy",
        title=f"ECE curve for {model_name} ({run_name}) on {dataset_name}",
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

    return float(ece)


def plot_sparsification_curve(
    targets, z_mu, z_sigma, path, model_name, dataset_name, run_name
):
    
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(z_mu)
    dist, idx = neigh.kneighbors(z_mu)
    
    # remove itself from
    dist = dist[:, 1]
    idx = idx[:, 1]
    
    preds = targets[idx]
    correct = (preds == targets).float()
    # query variance + database variance
    covar = (z_sigma + z_sigma[idx]).mean(dim=1)
    
    _, indices = torch.sort(-covar)
    
    accuracies = []
    for i in range(100):
        n = int(len(indices) * i / 100)
        accuracies.append(torch.mean(correct[indices[n:]]))
    
    accuracies = torch.stack(accuracies[:-1], dim=0).numpy()
    
    # Calculate AUSC (Area Under the Sparsification Curve)
    ausc = np.trapz(accuracies, dx=1 / len(accuracies))

    # Plot sparsification curve
    fig, ax = plt.subplots()
    ax.plot(accuracies)

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
    }
    with open(path / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f)

    return float(ausc)
