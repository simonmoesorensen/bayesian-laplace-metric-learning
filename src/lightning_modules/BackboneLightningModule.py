import datetime
import logging
import torch
from matplotlib import pyplot as plt
from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class BackboneLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)
        
        self.n_train_samples = 1
        self.n_val_samples = 1
        self.n_test_samples = 1

    def train_step(self, X, pairs, class_labels):
        
        z = self.forward(X)

        loss = self.loss_fn(z, None, indices_tuple=pairs)
        
        return z, loss

    def val_step(self, X, y, n_samples=1):
        z = self.forward(X)

        hard_pairs = self.miner(z, y)

        loss = self.loss_fn(z, y, indices_tuple=hard_pairs)
        
        return z, None, None

    def test_step(self, X, y, n_samples=1):
        z = self.forward(X)
        return z, None, None