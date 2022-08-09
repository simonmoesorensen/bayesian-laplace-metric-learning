
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image

from src import data, models
from src.laplace.metric_learning import train_metric
from src.laplace.post_hoc import post_hoc
from src.laplace.evaluate import evaluate_laplace
from src.laplace.utils import test_model
from src.visualization.plot_ood import plot_ood
from src.visualization.plot_roc import compute_and_plot_roc_curves


def run(latent_dim):
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # method = "post_hoc"

    # latent_dim = 32
    epochs = 30
    lr = 3e-4
    margin = 0.2
    normalize_encoding = False

    model = models.ConvNet(latent_dim, normalize_encoding).to(device)
    inference_model = model.linear
    # model = resnet50(num_classes=latent_dim, pretrained=False).to(device)
    # inference_model = nn.Sequential(model.fc)

    # ----------------------------------------------------------------------------------------------------------------------

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", 128)
    id_module.setup()
    train_loader = id_module.train_dataloader()
    id_loader = id_module.test_dataloader()

    logging.info("Finding MAP solution.")
    train_metric(model, train_loader, epochs, lr, margin, device)
    # torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict.pt")
    # # torch.save(model.state_dict(), f"pretrained/post_hoc/{id_label}/state_dict_normalized.pt")
    # model.load_state_dict(torch.load(f"pretrained/post_hoc/{id_label}/state_dict.pt"))
    state_dict = model.state_dict().copy()

    k = 10
    results = test_model(train_loader.dataset, id_loader.dataset, model, device, k=k)
    logging.info(f"MAP MAP@{k}: {results['mean_average_precision']:.2f}")
    logging.info(f"MAP Accuracy: {100*results['precision_at_1']:.2f}%")

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", 32)
    id_module.setup()
    train_loader = id_module.train_dataloader()

    model.load_state_dict(state_dict)
    mu_q, sigma_q = post_hoc(model, inference_model, train_loader, margin, device)
    # torch.save(mu_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_mu.pt")
    # torch.save(sigma_q.detach().cpu(), f"pretrained/post_hoc/{id_label}/laplace_sigma.pt")

    id_module = data.FashionMNISTDataModule("/work3/s174433/datasets", 512)
    id_module.setup()
    id_loader = id_module.test_dataloader()

    model.load_state_dict(state_dict)
    mean_id, variance_id = evaluate_laplace(model, inference_model, id_loader, mu_q, sigma_q, device)
    mean_id = mean_id.detach().cpu().numpy()
    variance_id = variance_id.detach().cpu().numpy()
    # np.save(f"results/{method}/{id_label}/id_laplace_mu.npy", mean_id.numpy())
    # np.save(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy", variance_id.numpy())

    # high_var_indices = torch.topk(variance_id.mean(dim=1), k=5, largest=True).indices
    # high_var_images = torch.stack([id_loader.dataset[i][0] for i in high_var_indices])
    # save_image(high_var_images, f"results/{method}/{id_label}/id_laplace_high_var.png", nrow=5)
    # low_var_indices = torch.topk(variance_id.mean(dim=1), k=5, largest=False).indices
    # low_var_images = torch.stack([id_loader.dataset[i][0] for i in low_var_indices])
    # save_image(low_var_images, f"results/{method}/{id_label}/id_laplace_low_var.png", nrow=5)

    ood_module = data.MNISTDataModule("/work3/s174433/datasets", 512, 4)
    ood_module.setup()
    ood_loader = ood_module.test_dataloader()

    model.load_state_dict(state_dict)
    mean_ood, variance_ood = evaluate_laplace(model, inference_model, ood_loader, mu_q, sigma_q, device)
    mean_ood = mean_ood.detach().cpu().numpy()
    variance_ood = variance_ood.detach().cpu().numpy()
    # np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy", mean_ood.numpy())
    # np.save(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy", variance_ood.numpy())

    # high_var_indices = torch.topk(variance_ood.mean(dim=1), k=5, largest=True).indices
    # high_var_images = torch.stack([ood_loader.dataset[i][0] for i in high_var_indices])
    # save_image(high_var_images, f"results/{method}/{id_label}/{ood_label}/ood_laplace_high_var.png", nrow=5)
    # low_var_indices = torch.topk(variance_ood.mean(dim=1), k=5, largest=False).indices
    # low_var_images = torch.stack([ood_loader.dataset[i][0] for i in low_var_indices])
    # save_image(low_var_images, f"results/{method}/{id_label}/{ood_label}/ood_laplace_low_var.png", nrow=5)

    fig, ax = plot_ood(mean_id, variance_id, mean_ood, variance_ood)
    fig.tight_layout()
    fig.savefig("ood_plot.png")

    metrics = compute_and_plot_roc_curves(".", variance_id, variance_ood)
    # print(metrics)
    # metrics = pd.DataFrame.from_dict({metric: [val] for metric, val in metrics.items()})
    # metrics_path = f"results/{method}/{id_label}/{ood_label}/ood_metrics.csv"
    # metrics.to_csv(metrics_path, index=False, header=True)
    return metrics["auroc"]


if __name__ == "__main__":
    print(run(2))

    # tmp = []
    # for i in range(5):
    #     auroc = run(2)
    #     tmp.append(auroc)
    # print(tmp)

    # auroc = {}
    # for d in range(2, 32+1, 2):
    #     tmp = []
    #     for i in range(3):
    #         tmp.append(run(d))
    #     auroc[d] = (np.mean(tmp), np.std(tmp))

    # print(auroc)
    # with open("auroc.pkl", "wb") as f:
    #     pickle.dump(auroc, f)

    # df = pd.DataFrame.from_dict(auroc, orient="index").reset_index()
    # df.columns = ["d", "auroc_mean", "auroc_std"]
    # print(df)

    # fig, ax = plt.subplots()
    # ax.errorbar(df["d"], df["auroc_mean"], yerr=df["auroc_std"], fmt="o")
    # ax.set_xlabel("Latent dimensions")
    # ax.set_ylabel("AUROC")
    # fig.tight_layout()
    # fig.savefig("auroc.png")
