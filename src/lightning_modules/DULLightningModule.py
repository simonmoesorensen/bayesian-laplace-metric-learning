import datetime
import logging
import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.lightning_modules.BaseLightningModule import BaseLightningModule

import torch.distributions as dist

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class DULLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        self.metrics.add("train_loss_kl")
        self.metrics.add("val_loss_kl")

    def epoch_start(self):
        self.metrics.reset(
            ["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"]
        )

    def epoch_end(self):
        self.log(["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"])

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

        self.metrics.update(f"{step}_loss", loss_backbone.item())
        self.metrics.update(f"{step}_loss_kl", loss_kl.item())

        return samples, loss

    def train_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        samples, loss = self.loss_step(mu_dul, std_dul, y, step="train")

        return samples, loss

    def val_start(self):
        self.metrics.reset(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

    def val_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        samples, _ = self.loss_step(mu_dul, std_dul, y, step="val")

        return mu_dul, std_dul, samples

    def val_end(self):
        self.log(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
            "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Validation MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("val_loss"),
                loss_KL=self.metrics.get("val_loss_kl"),
                acc=self.metrics.get("val_accuracy"),
                map_r=self.metrics.get("val_map_r"),
            ),
            flush=True,
        )

    def test_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        cov = torch.diag_embed(std_dul.square())
        pdist = dist.MultivariateNormal(mu_dul, cov)
        samples = pdist.rsample()

        return mu_dul, std_dul, samples

    def ood_step(self, X, y):
        return self.forward(X)

    def display(self, epoch, batch):
        tqdm.write("=" * 60)
        tqdm.write(
            "Epoch {}/{} Batch (Step) {}/{}\t"
            "Time {}\t"
            "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
            "Training Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Training MAP@r {map_r.val:.4f} ({map_r.avg:.4f})\t"
            "Lr {lr:.4f}".format(
                epoch + 1,
                self.args.num_epoch,
                batch + 1,
                len(self.train_loader) * self.args.num_epoch,
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("train_loss"),
                loss_KL=self.metrics.get("train_loss_kl"),
                acc=self.metrics.get("train_accuracy"),
                map_r=self.metrics.get("train_map_r"),
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )
