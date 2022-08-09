import datetime
import logging
import time
from pathlib import Path, PosixPath

import torch
import torch.distributions as tdist
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from src.lightning_modules.BaseLightningModule import BaseLightningModule

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class HIBLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        max_lr = 0.0003
        self.scheduler.max_lrs = self.scheduler._format_param(
            "max_lr", optimizer, max_lr
        )

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
        self.K = args.K

        # Move loss fn parameters to GPU
        self.loss_fn.cast_params(self.device)

    def optimizer_step(self):
        self.optimizer.step()
        self.loss_optimizer.step()

        self.loss_fn.apply(self.loss_fn.weight_clipper)

    def epoch_start(self):
        self.metrics.reset(
            ["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"]
        )

    def epoch_end(self):
        self.log(["train_loss", "train_loss_kl", "train_accuracy", "train_map_r"])

    def loss_step(self, mu, std, y, step):
        # Matrix of positive pairs
        pos_mask = y.view(-1, 1) == y.view(1, -1)

        # Don't sample diagonal (same images)
        pos_mask = pos_mask.fill_diagonal_(False)

        # Get lower triangular matrix to avoid duplicates
        pos_mask = torch.tril(pos_mask, -1)

        # Get indicies where matrix is true
        pos_idx = torch.nonzero(pos_mask)

        # Get negative pair indicies (not same image)
        neg_mask = pos_mask.fill_diagonal_(True)
        neg_idx = torch.nonzero(~neg_mask)

        # Random select n samples from neg_idx
        neg_idx = neg_idx[
            torch.randint(
                low=0,
                high=neg_idx.shape[0],
                size=(pos_idx.shape[0],),
                device=self.device,
            )
        ]

        # Get positive pairs from indices
        ap, pos = pos_idx.tensor_split(2, dim=1)
        an, neg = neg_idx.tensor_split(2, dim=1)
        ap, pos, an, neg = ap.view(-1), pos.view(-1), an.view(-1), neg.view(-1)

        # Create sample distribution
        cov = torch.diag_embed(std)
        pdist = tdist.MultivariateNormal(mu, cov)

        # Compare to unit gaussian r(z) ~ N(0, I)
        zdist = tdist.MultivariateNormal(
            torch.zeros_like(mu),
            torch.diag_embed(torch.ones_like(std)),
        )

        # Monte Carlo K times sampling, reparameterization trick in order
        # to do backprop
        # [K_samples, batch_size, embedding_space]
        samples = self.to_device(pdist.rsample([self.K]))

        # Repeat interleave tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[1,2],[3,4],[3,4],[4,5],[4,5]]
        k1_samples = samples.repeat_interleave(self.K, dim=0)

        # Repeat tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[3,4],[4,5],[1,2],[3,4],[4,5]]
        k2_samples = samples.repeat(self.K, 1, 1)

        # Convert 3D to 2D, we use self.K**3 because we have K_samples, repeated K times
        # Concatenates all K samples into one large
        # [batch_size * K, embedding_space] tensor
        k1_samples = k1_samples.view(
            self.K**2 * mu.shape[0], self.args.embedding_size
        )
        k2_samples = k2_samples.view(
            self.K**2 * mu.shape[0], self.args.embedding_size
        )

        # Repeat target to match the shape of the samples
        # [batch_size * K, embedding_space]
        y_k1 = y.repeat_interleave(self.K**2, dim=0)
        y_k2 = y.repeat(self.K**2, 1).view(-1)

        # See hib-pair-indicies.ipynb for more info
        # Scale the indices to match the shape of the samples
        def scale_indices(x):
            step = self.to_device(
                torch.arange(0, self.K**2).repeat_interleave(x.shape[0]) * mu.shape[0]
            )

            return x.repeat(self.K**2) + step

        # Scale indices to match repeated labels
        indices_tuple = [scale_indices(x) for x in [ap, pos, an, neg]]

        loss_soft_contrastive = self.loss_fn(
            embeddings=k1_samples,
            labels=y_k1,
            # something like this
            indices_tuple=indices_tuple,
            ref_emb=k2_samples,
            ref_labels=y_k2,
        )

        loss_kl = tdist.kl_divergence(pdist, zdist).sum() / mu.shape[0]

        loss = loss_soft_contrastive + self.args.kl_scale * loss_kl

        self.metrics.update(f"{step}_loss", loss_soft_contrastive.item())
        self.metrics.update(f"{step}_loss_kl", loss_kl.item())

        return samples, loss

    def train_step(self, X, y):
        # Pass images through the model
        mu, std = self.forward(X)

        samples, loss = self.loss_step(mu, std, y, step="train")

        return samples[0], loss

    def val_start(self):
        self.metrics.reset(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

    def val_step(self, X, y):
        mu, std = self.forward(X)

        samples, _ = self.loss_step(mu, std, y, step="val")

        return mu, std, samples[0]

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
        mu, std = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std)
        samples = mu + epsilon * std

        return mu, std, samples

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
            "Lr {lr:.4f}\t"
            "a {a:.4f}\t"
            "b {b:.4f}\t".format(
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
                a=self.loss_fn.A.data.item(),
                b=self.loss_fn.B.data.item(),
            )
        )

    def save_model(self, prefix=None):
        current_time = get_time()

        model_name = "Model_Epoch_{}_Time_{}_checkpoint.pth".format(
            self.epoch + 1, current_time
        )

        if prefix is not None:
            model_name = prefix + "_" + model_name

        path = Path(self.args.model_save_folder) / self.args.name
        model_path = path / model_name

        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model @ {str(path)}")
        self.save(content=self.model.module.module.state_dict(), filepath=str(model_path))


        loss_name = "Loss_Epoch_{}_Time_{}_checkpoint.pth".format(
            self.epoch + 1, current_time
        )

        loss_path = path / loss_name

        if prefix is not None:
            loss_name = prefix + "_" + loss_name

        print(f"Saving loss @ {str(path)}")
        torch.save(self.loss_fn, loss_path)