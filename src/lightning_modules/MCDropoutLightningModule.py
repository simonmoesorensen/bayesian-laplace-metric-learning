import datetime
import logging

from matplotlib import pyplot as plt
from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


class MCDropoutLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

    def forward(self, X, samples=100):
        return self.model(X, samples)

    def train_step(self, X, pairs):
        mu = self.forward(X, samples=None)

        loss = self.loss_fn(mu, pairs, indices_tuple=pairs)

        return mu, loss

    def val_step(self, X, y, n_samples=1):
        enable_dropout(self.model)

        mu, std, samples = self.forward(X)

        pairs = self.miner(mu, y)

        loss = self.loss_fn(mu, y, indices_tuple=pairs)

        return mu, std, samples

    def test_step(self, X, y, n_samples=1):
        enable_dropout(self.model)

        mu, std, samples = self.forward(X)

        return mu, std, samples
