import torch

from src.laplace.config import parse_args
from src.baselines.Backbone.models import Casia_Backbone, CIFAR10_Backbone, MNIST_Backbone
from src.data_modules import CasiaDataModule, CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule
from src.laplace.post_hoc import post_hoc


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

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
    
    model.load_state_dict(torch.load(args.model_path))

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        sampler=sampler,
    )

    device = "cuda"
    model.to(device)
    data_module.setup()

    inference_model = getattr(model, args.inference_model)

    mu_q, sigma_q = post_hoc(
        model, inference_model, data_module.train_dataloader(), margin=args.margin, device=device, method="full"
    )

    mu_id, var_id = evaluate_laplace(model, inference_model, data_module.test_dataloader(), mu_q, sigma_q, device)
    mu_id = mu_id.detach().cpu().numpy()
    var_id = var_id.detach().cpu().numpy()

    mu_ood, var_ood = evaluate_laplace(model, inference_model, data_module.ood_dataloader(), mu_q, sigma_q, device)
    mu_ood = mu_ood.detach().cpu().numpy()
    var_ood = var_ood.detach().cpu().numpy()

    fig, ax = plot_ood(mu_id, var_id, mu_ood, var_ood)
    fig.tight_layout()
    fig.savefig(f"ood_plot.png")


from typing import Tuple
import torch
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.distances import LpDistance
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
def evaluate_laplace(net, inference_net, loader, mu_q, sigma_q, device="cpu"):
    samples = sample_nn_weights(mu_q, sigma_q)

    pred_mean, pred_var = generate_predictions_from_samples_rolling(loader, samples, net, inference_net, device)
    return pred_mean, pred_var
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean, mode
from matplotlib.patches import Ellipse
import seaborn as sns
sns.set_theme(style="ticks")
c_id = "b"
c_ood = "r"
def plot_samples(mu, sigma_sq, latent1=0, latent2=1, limit=100, ax=None, color="b", label=None):
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
    
    print(f"mean={mean_sigma_sq.mean():.5f}, "
          f"std={mean_sigma_sq.std():.5f}, "
          f"min={mean_sigma_sq.min():.5f}, "
          f"max={mean_sigma_sq.max():.5f}")

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

if __name__ == "__main__":
    run(parse_args())
