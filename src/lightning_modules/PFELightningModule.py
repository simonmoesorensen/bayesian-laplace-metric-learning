import datetime
import logging

import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN
import torch.distributions as tdist

from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class PFELightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        # Avoid FAISS error on Ampere GPUs
        knn_func = CustomKNN(distances.LpDistance())

        max_lr = 0.0006
        self.scheduler.max_lrs = self.scheduler._format_param(
            "max_lr", optimizer, max_lr
        )

        # Metric calculation
        self.metric_calc = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k="max_bin_count",
            device=self.device,
            knn_func=knn_func,
        )

    def train_step(self, X, y):
        mu, std = self.forward(X)
        # pdist = torch.distributions.Normal(mu, std)
        cov = torch.stack([torch.diag(s) for s in (std)])
        pdist = tdist.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        pairs = self.miner(sample, y)

        var = std.square()

        
        # TODO switch out 128 with batch size and vectorize

        var_inv = (1/var).sum(axis=0).repeat(128,1)
        var_n = 1/var_inv

        mu_copy = mu.clone()
        for i in range(128):
            numerator = var_n[i,:].clone().repeat(128,1)
            mu_copy[i,:] = numerator.div(var.clone()).mul(mu.clone()).sum(axis=0)

        mu = mu_copy.clone()
        # var = var_n.clone()

        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        self.metrics.update("train_loss", loss.item())

        return sample, loss

    def val_step(self, X, y):
        mu, std = self.forward(X)

        # pdist = torch.distributions.Normal(mu, std)
        cov = torch.stack([torch.diag(s) for s in (std)])
        pdist = tdist.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        pairs = self.miner(sample, y)

        var = std.square()
        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        self.metrics.update("val_loss", loss.item())
        return mu, std, sample

    def test_step(self, X, y):
        mu, std = self.forward(X)

        pdist = torch.distributions.Normal(mu, std)
        sample = self.to_device(pdist.rsample())

        return mu, std, sample

    def ood_step(self, X, y):
        return self.forward(X)
