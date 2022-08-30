import json
import logging
from pathlib import Path
from unittest import result

import pandas as pd
import torch
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN
from torch.utils.data import DataLoader, TensorDataset
from laplace.post_hoc import evaluate_laplace, visualize

from src.laplace.config import parse_args
from src.baselines.Backbone.models import Casia_Backbone, CIFAR10_Backbone, MNIST_Backbone
from src.data_modules import CasiaDataModule, CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule
from src.laplace.post_hoc import post_hoc
from src.visualize import visualize_all
from src.evaluation.calibration_curve import calibration_curves
from src.recall_at_k import AccuracyRecall
from src.distances import ExpectedSquareL2Distance
from pytorch_metric_learning.distances import LpDistance
import gc 
from encodings import normalize_encoding
import logging
from tkinter import N

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.hessian.layerwise import ContrastiveHessianCalculator
def sample_neural_network_wights(parameters, posterior_scale, n_samples=32):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_online(epochs, freq, nn_samples, device, lr, train_loader, margin, net, net_inference):
    contrastive_loss = losses.ContrastiveLoss(neg_margin=margin)
    # miner = miners.MultiSimilarityMiner()
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.ALL,
        neg_strategy=miners.BatchEasyHardMiner.ALL)
    hessian_calculator = ContrastiveHessianCalculator(margin=margin, device=device)

    hessian_calculator.init_model(net_inference)

    num_params = sum(p.numel() for p in net_inference.parameters())
    dataset_size = len(train_loader.dataset)
    scale = 1.0
    prior_prec = 1.0

    optim = Adam(net.parameters(), lr=lr)

    h = 1e10 * torch.ones((num_params,), device=device)
    posterior_precision = h * scale + prior_prec
    sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)

    for epoch in (range(epochs)):
        epoch_losses = []
        compute_hessian = epoch % freq == 0

        if compute_hessian:
            hs = 0

        for i, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            mu_q = parameters_to_vector(net_inference.parameters())

            sampled_nn = sample_neural_network_wights(mu_q, sigma_q, n_samples=nn_samples)

            con_loss = 0
            if compute_hessian:
                h = 0

            for nn_i in sampled_nn:
                vector_to_parameters(nn_i, net_inference.parameters())
                output = net(x)
                hard_pairs = miner(output, y)

                if compute_hessian:
                    # Adjust hessian to the batch size
                    scaler = dataset_size**2 / (len(hard_pairs[0]) + len(hard_pairs[2]))**2
                    hessian_batch = hessian_calculator.compute_batch_pairs(hard_pairs)
                    hessian_batch = torch.clamp(hessian_batch, min=0)
                    h += hessian_batch * scaler

                con_loss += contrastive_loss(output, y, hard_pairs)

            if compute_hessian:
                h /= nn_samples

                hs += h
                H = hs / (i+1)
                posterior_precision = H * scale + prior_prec
                sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)
                # posterior_precision = h * scale + prior_prec
                # sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)
                # print(f"{sigma_q.mean()=}")

            
            con_loss /= nn_samples
            loss = con_loss
            vector_to_parameters(mu_q, net_inference.parameters())
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())

        loss_mean = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss_mean=} for {epoch=}")

    mu_q = parameters_to_vector(net_inference.parameters())
    posterior_precision = H * scale + prior_prec
    sigma_q = 1.0 / (posterior_precision.sqrt() + 1e-6)
    return mu_q, sigma_q


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

    mu_q, sigma_q = train_online(args.num_epoch, args.hessian_freq, args.posterior_samples, device, args.lr, data_module.train_dataloader(), args.margin, model, inference_model)
    torch.save(model.state_dict(), f"{pretrained_path}/state_dict_{args.embedding_size}.pt")
    torch.save(mu_q, f"{pretrained_path}/mu_q_{args.embedding_size}.pt")
    torch.save(sigma_q, f"{pretrained_path}/sigma_q_{args.embedding_size}.pt")
    # model.load_state_dict(torch.load(f"{pretrained_path}/state_dict_{args.embedding_size}.pt", map_location=device))
    # mu_q = torch.load(f"{pretrained_path}/mu_q_{args.embedding_size}.pt", map_location=device)
    # sigma_q = torch.load(f"{pretrained_path}/sigma_q_{args.embedding_size}.pt", map_location=device)

    mu_id, var_id = evaluate_laplace(model, inference_model, data_module.test_dataloader(), mu_q, sigma_q, device)
    mu_id = mu_id.detach().cpu()#.numpy()
    var_id = var_id.detach().cpu()#.numpy()
    id_images = torch.cat([x for x, _ in data_module.test_dataloader()], dim=0).detach().cpu()#.numpy()
    id_labels = torch.cat([y for _, y in data_module.test_dataloader()], dim=0).detach().cpu()#.numpy()

    mu_ood, var_ood = evaluate_laplace(model, inference_model, data_module.ood_dataloader(), mu_q, sigma_q, device)
    mu_ood = mu_ood.detach().cpu()#.numpy()
    var_ood = var_ood.detach().cpu()#.numpy()
    ood_images = torch.cat([x for x, _ in data_module.ood_dataloader()], dim=0).detach().cpu()#.numpy()

    mu_train, var_train = evaluate_laplace(model, inference_model, data_module.train_dataloader(), mu_q, sigma_q, device)
    mu_train = mu_train.detach().cpu()#.numpy()
    var_train = var_train.detach().cpu()#.numpy()
    train_labels = torch.cat([y for _, y in data_module.train_dataloader()], dim=0).detach().cpu()#.numpy()
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
    pd.DataFrame.from_dict(results, orient="index").assign(dim=args.embedding_size).to_csv(f"{figure_path}/metrics.csv", mode="a", header=False)
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
    pd.DataFrame.from_dict({
        "mean_average_precision_expected": results["mean_average_precision"],
        "precision_at_1_expected": results["precision_at_1"],
        "recall_at_k_expected": results["recall_at_k"],
        }, orient="index")\
        .assign(dim=args.embedding_size)\
        .to_csv(f"{figure_path}/metrics.csv", mode="a", header=False)


    visualize(mu_id, var_id, id_images, id_labels, mu_ood, var_ood, ood_images, args.dataset, Path(figure_path))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(parse_args())
