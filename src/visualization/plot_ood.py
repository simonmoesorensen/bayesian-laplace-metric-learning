import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from matplotlib.patches import Ellipse
import seaborn as sns

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


def plot_ood(mu_id, var_id, mu_ood, var_ood):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_samples(mu_id, var_id, limit=100, color=c_id, label="ID", ax=ax[0])
    plot_histogram(var_id, color=c_id, ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color=c_ood, label="OOD", ax=ax[0])
    plot_histogram(var_ood, color=c_ood, ax=ax[1])
    ax[0].legend()
    return fig, ax


if __name__ == "__main__":
    id_label = "cifar10"
    id_title = "CIFAR-10"

    # ood_label = "cifar100"
    # ood_title = "CIFAR-100"
    ood_label = "svhn"
    ood_title = "SVHN"
    # ood_label = "noise"
    # ood_title = "Noise"

    mu_id = np.load(f"results/laplace/{id_label}/id_laplace_mu.npy")
    var_id = np.load(f"results/laplace/{id_label}/id_laplace_sigma_sq.npy")
    mu_ood = np.load(f"results/laplace/{id_label}/{ood_label}/ood_laplace_mu.npy")
    var_ood = np.load(f"results/laplace/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy")

    fig, ax = plot_ood(mu_id, var_id, mu_ood, var_ood)
    fig.suptitle(f"Trained on {id_title}, OOD {ood_title}")
    fig.savefig(f"figures/ood_plot_{id_label}_{ood_label}.png")
