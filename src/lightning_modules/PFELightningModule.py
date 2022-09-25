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
        #TODO: maybe I introduced a bug here. It seems like they are calculating the entire matric
        # and then indexing into it. If this is the case, then it should be fine. I think.
        # although, it seems a bit expensive...
        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)

        return mu, loss

    def val_step(self, X, y, n_samples=1):
        mu, std = self.forward(X)
        var = std ** 2
        cov = torch.diag_embed(var)
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        samples = pdist.rsample([n_samples])
        
        # project into unit sphere
        samples = samples / samples.norm(dim=1, keepdim=True)

        panc, pos, _, _ = self.miner(samples, y)
        pairs = (panc, pos, [], [])

        loss = self.loss_fn(embeddings=mu, ref_emb=var, indices_tuple=pairs)
        
        return mu, std, samples

    def test_step(self, X, y, n_samples=1):
        mu, std = self.forward(X)
        var = std ** 2
        cov = torch.diag_embed(var)
        pdist = torch.distributions.MultivariateNormal(mu, cov)
        samples = pdist.rsample([n_samples])
        
        # project into unit sphere
        samples = samples / samples.norm(dim=1, keepdim=True)
        
        return mu, std, samples
