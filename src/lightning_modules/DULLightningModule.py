import datetime
import logging
import time

import torch
import torch.distributions as dist
from src.lightning_modules.BaseLightningModule import BaseLightningModule
from tqdm import tqdm

logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class DULLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        self.loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)

    def optimizer_step(self):
        super().optimizer_step()
        self.loss_optimizer.step()

    def loss_step(self, mu, std, y, step):
        variance_dul = std.square()

        cov = torch.diag_embed(variance_dul)
        pdist = dist.MultivariateNormal(mu, cov)
        samples = pdist.rsample()

        hard_pairs = self.miner(samples, y)
        loss_backbone = self.loss_fn(samples, y, hard_pairs)

        loss_kl = (
            ((variance_dul + mu.square() - torch.log(variance_dul) - 1) * 0.5)
            .sum(dim=-1)
            .mean()
        )

        loss = loss_backbone + self.args.kl_scale * loss_kl

        return samples, loss

    def train_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        samples, loss = self.loss_step(mu_dul, std_dul, y, step="train")

        return samples, loss

    def val_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        samples, _ = self.loss_step(mu_dul, std_dul, y, step="val")

        return mu_dul, std_dul, samples

    def test_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        cov = torch.diag_embed(std_dul ** 2)
        pdist = dist.MultivariateNormal(mu_dul, cov)
        samples = pdist.rsample()

        return mu_dul, std_dul, samples

    def ood_step(self, X, y):
        return self.forward(X)
