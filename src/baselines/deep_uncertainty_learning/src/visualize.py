import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from matplotlib.patches import Ellipse
import seaborn as sns
import torch
import torchmetrics
import json
import pandas as pd

sns.set()

c_id = "b"
c_ood = "r"


def plot_samples(mu, sigma_sq, latent1=0, latent2=1, limit=100, ax=None, color="b", label=None):
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


def plot_histogram(sigma_sq, mean="harmonic", ax=None, color="b", label=None):
    if ax is None:
        _, ax = plt.subplots()

    if mean == "harmonic":
        mean_sigma_sq = hmean(sigma_sq, axis=1)
    elif mean == "arithmetic":
        mean_sigma_sq = np.mean(sigma_sq, axis=1)
    else:
        raise NotImplementedError

    sns.kdeplot(mean_sigma_sq, ax=ax, color=color, label=label)
    ax.set(xlabel="Variance")


def plot_ood(mu_id, var_id, mu_ood, var_ood, vis_path, prefix):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_samples(mu_id, var_id, limit=100, color=c_id, label="ID", ax=ax[0])
    plot_histogram(var_id, color=c_id, ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color=c_ood, label="OOD", ax=ax[0])
    plot_histogram(var_ood, color=c_ood, ax=ax[1])
    ax[0].legend()
    fig.suptitle("ID vs OOD comparison")
    fig.savefig(vis_path / f"{prefix}ood_comparison.png")
    return fig, ax


def plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix):
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
    plt.title('OOD ROC Curve')
    plt.legend()
    fig.savefig(vis_path / f"{prefix}ood_roc_curve.png")
    plt.cla()
    plt.close()

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
    plt.title('OOD Precision-Recall Curve')
    plt.legend()
    fig.savefig(vis_path / f"{prefix}ood_precision_recall_curve.png")
    plt.cla()
    plt.close()

    metrics = {}

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)
    metrics["auprc"] = float(auprc_score.numpy())

    # compute false positive rate at 80
    num_id = len(id_sigma)

    for p in range(0, 100, 10):
        # if there is no difference in variance
        try:
            metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
        except:
            metrics[f"fpr{p}"] = "none"
        else:
            continue

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )
    metrics["auroc"] = float(auroc_score.numpy())
    print(f"Metrics: {metrics}")
    
    # save metrics
    with open(vis_path / "ood_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)