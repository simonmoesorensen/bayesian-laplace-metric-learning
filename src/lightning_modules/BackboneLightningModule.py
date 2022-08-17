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

        self.to_visualize = False

        # self.loss_optimizer = optim.SGD(loss_fn.parameters(), lr=0.01)

    # def optimizer_step(self):
    #     self.loss_optimizer.step()
    #     return super().optimizer_step()

    def train_step(self, X, y):
        z = self.forward(X)

        hard_pairs = self.miner(z, y)

        loss = self.loss_fn(z, y, indices_tuple=hard_pairs)

        self.metrics.update("train_loss", loss.item())

        return z, loss

    def val_step(self, X, y):
        z = self.forward(X)

        hard_pairs = self.miner(z, y)

        loss = self.loss_fn(z, y, indices_tuple=hard_pairs)

        self.metrics.update("val_loss", loss.item())
        return 0, 0, z

    def test_step(self, X, y):
        z = self.forward(X)
        return 0, 0, z

    def ood_step(self, X, y):
        raise ValueError("Backbone module is not probabilistic")
