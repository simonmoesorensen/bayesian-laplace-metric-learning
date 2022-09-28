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

    def loss_step(self, mu, std, class_labels):
        variance_dul = std.square()
        
        z = mu + std * torch.randn_like(std)
        loglik = self.loss_fn(z, class_labels, None)

        loss_kl = (- 0.5 * torch.sum(1 + torch.log(variance_dul) - mu.square() - variance_dul)).mean()
        loss = loglik + self.args.kl_scale * loss_kl
        
        return loss

    def train_step(self, X, pairs, class_labels):
        mu_dul, std_dul = self.forward(X)

        loss = self.loss_step(mu_dul, std_dul, class_labels)
        
        return mu_dul, loss

    def val_step(self, X, y, n_samples=1):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        cov = torch.diag_embed(std_dul ** 2)
        pdist = dist.MultivariateNormal(mu_dul, cov)
        samples = pdist.rsample([n_samples])

        return mu_dul, std_dul, samples

    def test_step(self, X, y, n_samples=1):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        cov = torch.diag_embed(std_dul ** 2)
        pdist = dist.MultivariateNormal(mu_dul, cov)
        samples = pdist.rsample([n_samples])

        return mu_dul, std_dul, samples