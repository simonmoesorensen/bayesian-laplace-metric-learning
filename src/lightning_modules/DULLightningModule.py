import datetime
import logging
import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN

from src.lightning_modules.BaseLightningModule import BaseLightningModule

from src.utils import l2_norm

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class DULLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        # REQUIRED FOR ARCFACE LOSS
        self.loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)

        # Required for ARCFACE loss
        knn_func = CustomKNN(CosineSimilarity())

        # Metric calculation
        self.metric_calc = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k="max_bin_count",
            device=self.device,
            knn_func=knn_func,
        )

        self.metrics.add("train_loss_kl")
        self.metrics.add("val_loss_kl")

    def optimizer_step(self):
        self.optimizer.step()
        self.loss_optimizer.step()

    def epoch_start(self):
        self.metrics.reset(
            ["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"]
        )

    def epoch_end(self):
        self.log(["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"])


    def train_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        epsilon = torch.randn_like(std_dul)
        samples = mu_dul + epsilon * std_dul
        variance_dul = std_dul**2

        norm_samples = l2_norm(samples)

        hard_pairs = self.miner(norm_samples, y)
        loss_backbone = self.loss_fn(norm_samples, y, hard_pairs)

        loss_kl = (
            ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
            .sum(dim=-1)
            .mean()
        )

        loss = loss_backbone + self.args.kl_scale * loss_kl

        self.metrics.update("train_loss", loss_backbone.item())
        self.metrics.update("train_loss_kl", loss_kl.item())

        return norm_samples, loss

    def val_start(self):
        self.metrics.reset(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

    def val_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std_dul)
        samples = mu_dul + epsilon * std_dul
        variance_dul = std_dul**2

        norm_samples = l2_norm(samples)

        hard_pairs = self.miner(norm_samples, y)
        loss = self.loss_fn(norm_samples, y, hard_pairs)

        loss_kl = (
            ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
            .sum(dim=-1)
            .mean()
        )

        self.metrics.update("val_loss", loss.item())
        self.metrics.update("val_loss_kl", loss_kl.item())
        return mu_dul, std_dul, norm_samples

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
                loss=self.metrics.get('val_loss'),
                loss_KL=self.metrics.get('val_loss_kl'),
                acc=self.metrics.get('val_accuracy'),
                map_r=self.metrics.get('val_map_r'),
            ),
            flush=True,
        )
    
    def test_step(self, X, y):
        mu_dul, std_dul = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std_dul)
        samples = mu_dul + epsilon * std_dul

        norm_samples = l2_norm(samples)

        return mu_dul, std_dul, norm_samples

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
            "Training MAP@r {map_r.val:.4f} ({map_r.avg:.4f})".format(
                epoch + 1,
                self.args.num_epoch,
                batch + 1,
                len(self.train_loader) * self.args.num_epoch,
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("train_loss"),
                loss_KL=self.metrics.get("train_loss_kl"),
                acc=self.metrics.get("train_accuracy"),
                map_r=self.metrics.get("train_map_r"),
            )
        )
