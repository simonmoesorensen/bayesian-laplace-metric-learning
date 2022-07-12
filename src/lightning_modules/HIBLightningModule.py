import datetime
import logging
import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from src.lightning_modules.BaseLightningModule import BaseLightningModule

from src.utils import l2_norm

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class HIBLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        # REQUIRED FOR SOFT CONTRASTIVE LOSS
        self.loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)

        # Required for SOFT CONTRASTIVE loss
        knn_func = CustomKNN(LpDistance())

        # Metric calculation
        self.metric_calc = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k="max_bin_count",
            device=self.device,
            knn_func=knn_func,
        )

        self.metrics.add("train_loss_kl")
        self.metrics.add("val_loss_kl")

        # Monte Carlo K times sampling
        K = 8

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
        # Get mined pairs
        mu, std = self.forward(X)

        pos_anc, pos, neg_anc, neg = self.miner(mu, y)
        
        print(pairs)
        for x1, x2 in pairs:
            # Pass to forward pass (x1, x2)
            # [1, embedding_size]
            mu_x1, std_x1 = self.forward(x1)
            mu_x2, std_x2 = self.forward(x2)

            # Reparameterization trick, sample K times
            epsilon1 = torch.randn(std_x1.shape[0], self.K)
            epsilon2 = torch.randn(std_x2.shape[0], self.K)

            samples_x1 = mu_x1 + epsilon1 * std_x1
            samples_x2 = mu_x2 + epsilon2 * std_x2


            

        # Get the normal distributions from x1 (x1_mu, x1_std) and x2 (x2_mu, x2_std)

        # Compute the K-squared distance samples 

        # Give k-squared distance samples to the loss function

        mu, std = self.forward(X)

        epsilon = torch.randn_like(std)
        samples = mu + epsilon * std
        variance = std**2

        norm_samples = l2_norm(samples)

        loss_kl = (
            ((variance + mu**2 - torch.log(variance) - 1) * 0.5)
            .sum(dim=-1)
            .mean()
        )

        loss_soft_constrastive = self.loss_fn(norm_samples, y)

        loss = loss_soft_constrastive + self.args.kl_scale * loss_kl

        self.metrics.update("train_loss", loss_soft_constrastive.item())
        self.metrics.update("train_loss_kl", loss_kl.item())

        return norm_samples, loss

    def val_start(self):
        self.metrics.reset(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

    def val_step(self, X, y):
        mu, std = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std)
        samples = mu + epsilon * std
        variance = std**2

        norm_samples = l2_norm(samples)

        loss_kl = (
            ((variance + mu**2 - torch.log(variance) - 1) * 0.5)
            .sum(dim=-1)
            .mean()
        )

        loss_soft_contrastive = self.loss_fn(norm_samples, y)

        self.metrics.update("val_loss", loss_soft_contrastive.item())
        self.metrics.update("val_loss_kl", loss_kl.item())

        return mu, std, norm_samples
    
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
        mu, std = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std)
        samples = mu + epsilon * std

        norm_samples = l2_norm(samples)

        return mu, std, norm_samples

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
