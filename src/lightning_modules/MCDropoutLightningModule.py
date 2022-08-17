import datetime
import logging

import torch
from matplotlib import pyplot as plt

from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class MCDropoutLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

    def forward(self, X, samples=100):
        return self.model(X, samples)

    def train_step(self, X, y):
        mu, _ = self.forward(X, samples=None)

        pairs = self.miner(mu, y)

        loss = self.loss_fn(mu, y, indices_tuple=pairs)

        self.metrics.update("train_loss", loss.item())

        return mu, loss

    def val_step(self, X, y):
        enable_dropout(self.model)

        mu, std = self.forward(X)

        cov = torch.diag_embed(std.square())
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        pairs = self.miner(sample, y)

        loss = self.loss_fn(mu, y, indices_tuple=pairs)

        self.metrics.update("val_loss", loss.item())
        return mu, std, sample

    def test_step(self, X, y):
        enable_dropout(self.model)

        mu, std = self.forward(X)

        cov = torch.diag_embed(std.square())
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        return mu, std, sample

    def ood_step(self, X, y):
        enable_dropout(self.model)

        return self.forward(X)
