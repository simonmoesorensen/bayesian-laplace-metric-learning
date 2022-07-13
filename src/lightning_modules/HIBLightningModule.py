import datetime
import logging
import time

import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from src.baselines.HIB.losses import WeightClipper
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
        self.K = args.K

        # KL Divergence
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

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

    def train_step(self, X, y):
        # Pass images through the model
        mu, std = self.forward(X)

        epsilon = torch.randn(self.K, std.shape[0], std.shape[1], device=self.device)
        # [K_samples, batch_size, embedding_space]
        samples = mu + std * epsilon
        samples = l2_norm(samples)

        pair_indices = self.miner(samples[0], y)

        # Repeat interleave tensor so something like [[1,2],[3,4],[4,5]] becomes 
        # [[1,2],[1,2],[3,4],[3,4],[4,5],[4,5]]
        k1_samples = samples.repeat_interleave(self.K**2, dim=0)

        # Repeat tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[3,4],[4,5],[1,2],[3,4],[4,5]]
        k2_samples = samples.repeat(self.K**2, 1, 1)
            
        # Convert 3D to 2D, we use self.K**3 because we have K_samples, repeated K**2 times
        # Concatenates all K samples into one large [batch_size * K, embedding_space] tensor
        k1_samples = k1_samples.view(self.K**3 * X.shape[0], self.args.embedding_size)
        k2_samples = k2_samples.view(self.K**3 * X.shape[0], self.args.embedding_size)
        
        # Repeat target to match the shape of the samples [batch_size * K, embedding_space]
        y_k1 = y.repeat_interleave(self.K**3, dim=0)
        y_k2 = y.repeat(self.K**3, 1).view(-1)

        pos_anc, pos, neg_anc, neg = pair_indices

        # Scale the indices to match the shape of the samples
        scale_indices = lambda x: (
                x.repeat(self.K, 1).T + (torch.arange(0, self.K, device=self.device) * self.batch_size)
            ).T.view(-1)

        pos_anc_scaled = scale_indices(pos_anc)
        pos_scaled  = scale_indices(pos)
        neg_anc_scaled  = scale_indices(neg_anc)
        neg_scaled  = scale_indices(neg)

        loss_soft_constrastive = self.loss_fn(
            embeddings=k1_samples,
            labels=y_k1,
            # something like this
            indices_tuple=(pos_anc_scaled, pos_scaled , neg_anc_scaled , neg_scaled ),
            ref_emb=k2_samples,
            ref_labels=y_k2,
        )

        loss_kl = torch.Tensor([self.kl_loss(sample.T, y) for sample in samples]).mean()

        loss = loss_soft_constrastive + self.args.kl_scale * loss_kl

        self.metrics.update("train_loss", loss_soft_constrastive.item())
        self.metrics.update("train_loss_kl", loss_kl.item())

        return samples[0], loss

    def val_start(self):
        self.metrics.reset(["val_loss", "val_loss_kl", "val_accuracy", "val_map_r"])

    def val_step(self, X, y):
        mu, std = self.forward(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std)
        samples = mu + epsilon * std
        norm_samples = l2_norm(samples)

        pair_indices = self.miner(samples[0], y)

        # Repeat interleave tensor so something like [[1,2],[3,4],[4,5]] becomes 
        # [[1,2],[1,2],[3,4],[3,4],[4,5],[4,5]]
        k1_samples = samples.repeat_interleave(self.K**2, dim=0)

        # Repeat tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[3,4],[4,5],[1,2],[3,4],[4,5]]
        k2_samples = samples.repeat(self.K**2, 1, 1)
            
        # Convert 3D to 2D, we use self.K**3 because we have K_samples, repeated K**2 times
        # Concatenates all K samples into one large [batch_size * K, embedding_space] tensor
        k1_samples = k1_samples.view(self.K**3 * X.shape[0], self.args.embedding_size)
        k2_samples = k2_samples.view(self.K**3 * X.shape[0], self.args.embedding_size)
        
        # Repeat target to match the shape of the samples [batch_size * K, embedding_space]
        y_k1 = y.repeat_interleave(self.K**3, dim=0)
        y_k2 = y.repeat(self.K**3, 1).view(-1)

        pos_anc, pos, neg_anc, neg = pair_indices

        # Scale the indices to match the shape of the samples
        scale_indices = lambda x: (
                x.repeat(self.K, 1).T + (torch.arange(0, self.K, device=self.device) * self.batch_size)
            ).T.view(-1)

        pos_anc_scaled = scale_indices(pos_anc)
        pos_scaled  = scale_indices(pos)
        neg_anc_scaled  = scale_indices(neg_anc)
        neg_scaled  = scale_indices(neg)

        loss_soft_constrastive = self.loss_fn(
            embeddings=k1_samples,
            labels=y_k1,
            # something like this
            indices_tuple=(pos_anc_scaled, pos_scaled , neg_anc_scaled , neg_scaled ),
            ref_emb=k2_samples,
            ref_labels=y_k2,
        )

        loss_kl = torch.Tensor([self.kl_loss(sample.T, y) for sample in samples]).mean()

        

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