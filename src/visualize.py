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


def visualize_top_5(
    id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix, n=5
):
    """Visualize the top 5 highest and lowest variance images"""
    model_name, dataset, run_name = get_names(vis_path)

    # Get colormap
    channels = id_images.shape[1]

    if channels == 1:
        cmap = "gray"
    elif channels == 3:
        cmap = None

    # Get l2 norm of ID variances
    id_sigma_mu = hmean(id_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of ID variances
    top_5_id = (-id_sigma_mu).argsort()[:n]
    bot_5_id = (-id_sigma_mu).argsort()[-n:]

    # Get l2 norm of OOD variances
    ood_sigma_mu = hmean(ood_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of OOD variances
    top_5_ood = (-ood_sigma_mu).argsort()[:n]
    bot_5_ood = (-ood_sigma_mu).argsort()[-n:]

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
        plt.title(f"ID V={id_sigma_mu[top_5_id[col]]:.2E}")
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
        plt.title(f"ID V={id_sigma_mu[bot_5_id[col]]:.2E}")
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
        plt.title(f"OOD V={ood_sigma_mu[top_5_ood[col]]:.2E}")
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
        plt.title(f"OOD V={ood_sigma_mu[bot_5_ood[col]]:.2E}")
        if col == 0:
            plt.ylabel("Bot 5 var OOD")
        counter += 1

    plt.suptitle(
        f"Top and bottom 5 variance images for model {model_name} ({run_name}) on dataset {dataset}"
    )

    fig.savefig(vis_path / f"{prefix}top_bot_5_var.png")


def visualize_all(
    id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
):
    id_sigma = torch.cat(id_sigma, dim=0).cpu().detach().numpy()
    id_mu = torch.cat(id_mu, dim=0).cpu().detach().numpy()
    ood_sigma = torch.cat(ood_sigma, dim=0).cpu().detach().numpy()
    ood_mu = torch.cat(ood_mu, dim=0).cpu().detach().numpy()
    id_images = torch.cat(id_images, dim=0).cpu().detach().numpy()
    ood_images = torch.cat(ood_images, dim=0).cpu().detach().numpy()

    if not prefix.endswith("_"):
        prefix += "_"

    # visualize top 5 and bottom 5 variance images
    visualize_top_5(
        id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
    )

    # Visualize
    plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix)

    id_var = id_sigma**2
    ood_var = ood_sigma**2
    plot_ood(id_mu, id_var, ood_mu, ood_var, vis_path, prefix)


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
    model_name, dataset, run_name = get_names(vis_path)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_samples(mu_id, var_id, limit=100, color=c_id, label="ID", ax=ax[0])
    plot_histogram(var_id, color=c_id, ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color=c_ood, label="OOD", ax=ax[0])
    plot_histogram(var_ood, color=c_ood, ax=ax[1])
    ax[0].legend()
    fig.suptitle(f"ID vs OOD comparison for model {model_name} on dataset {dataset}")
    fig.savefig(vis_path / f"{prefix}ood_comparison.png")
    return fig, ax


def plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix):
    model_name, dataset, run_name = get_names(vis_path)
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
    plt.title(f"OOD Precision-Recall Curve for model {model_name} ({run_name}) on dataset {dataset}")
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


def get_names(vis_path):
    # returns (model name, dataset trained on)
    return (vis_path.parts[-5], vis_path.parts[-3], vis_path.parts[-2])
