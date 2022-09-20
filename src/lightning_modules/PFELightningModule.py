import datetime
import logging

import torch
from matplotlib import pyplot as plt
from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class PFELightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

    def train_step(self, x, pairs):
                
        mu, std = self.forward(x)
        
        var = std ** 2
        cov = torch.diag_embed(var)
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        self.metrics.update("train_loss", loss.item())

        return sample, loss

    def val_step(self, X, y):
        mu, std = self.forward(X)
        var = std ** 2
        cov = torch.diag_embed(var)
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())

        panc, pos, _, _ = self.miner(sample, y)
        pairs = (panc, pos, [], [])

        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        self.metrics.update("val_loss", loss.item())
                
        return mu, std, sample.unsqueeze(0)

    def test_step(self, X, y):
        mu, std = self.forward(X)
        var = std ** 2
        cov = torch.diag_embed(var)
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        sample = self.to_device(pdist.rsample())
        
        return mu, std, sample.unsqueeze(0)

    def ood_step(self, X, y):
        return self.forward(X)
