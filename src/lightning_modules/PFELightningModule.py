import datetime
import logging

import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN

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

        # Metric calculation
        self.metric_calc = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k="max_bin_count",
            device=self.device,
            knn_func=knn_func,
        )

    def train_step(self, X, y):
        mu, std = self.forward(X)
        pdist = torch.distributions.Normal(mu, std)
        sample = self.to_device(pdist.rsample())

        pairs = self.miner(sample, y)

        var = std.square()
        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        self.metrics.update("train_loss", loss.item())

        return sample, loss

    def val_step(self, X, y):
        mu, std = self.forward(X)

        pdist = torch.distributions.Normal(mu, std)
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
