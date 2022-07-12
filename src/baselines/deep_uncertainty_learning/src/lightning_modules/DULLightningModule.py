import datetime
import logging
from pathlib import Path, PosixPath
from re import I
import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN
from lightning_modules.BaseLightningModule import BaseLightningModule

from utils import l2_norm

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class DULLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args, to_visualize=False):
        super().init(model, loss_fn, miner, optimizer, args, to_visualize)

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
        mu_dul, std_dul = self.model(X)

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
        mu_dul, std_dul = self.model(X)

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
        mu_dul, std_dul = self.model(X)

        # Reparameterization trick
        epsilon = torch.randn_like(std_dul)
        samples = mu_dul + epsilon * std_dul

        norm_samples = l2_norm(samples)

        return mu_dul, std_dul, norm_samples

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

    # # noinspection PyMethodOverriding
    # def run(self):

    #     print(f"Training")
    #     self.model.train()

    #     if not self.name:
    #         raise ValueError("Please run .init()")

    #     losses = AverageMeter()
    #     losses_KL = AverageMeter()
    #     self.train_accuracy = AverageMeter()
    #     self.train_map_r = AverageMeter()

    #     DISP_FREQ = len(self.train_loader) // 20  # frequency to display training loss
    #     batch = 0

    #     for epoch in range(self.args.num_epoch):
    #         losses.reset()
    #         losses_KL.reset()
    #         self.train_accuracy.reset()
    #         self.train_map_r.reset()

    #         if epoch < self.args.resume_epoch:
    #             continue

    #         self.epoch = epoch

    #         for image, target in tqdm(self.train_loader, desc='Training'):
    #             self.optimizer.zero_grad()

    #             mu_dul, std_dul = self.model(image)

    #             epsilon = torch.randn_like(std_dul)
    #             samples = mu_dul + epsilon * std_dul
    #             variance_dul = std_dul**2

    #             norm_samples = l2_norm(samples)

    #             hard_pairs = self.miner(norm_samples, target)
    #             loss_backbone = self.loss_fn(norm_samples, target, hard_pairs)

    #             loss_kl = (
    #                 ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
    #                 .sum(dim=-1)
    #                 .mean()
    #             )

    #             loss_backbone += self.args.kl_scale * loss_kl

    #             self.backward(loss_backbone)
    #             self.optimizer.step()
    #             self.loss_optimizer.step()

    #             losses_KL.update(loss_kl.item(), image.size(0))
    #             losses.update(loss_backbone.data.item(), image.size(0))

    #             # dispaly training loss & acc every DISP_FREQ
    #             if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
    #                 # Metrics
    #                 metrics = self.metrics.get_accuracy(
    #                     query=norm_samples,
    #                     reference=norm_samples,
    #                     query_labels=target,
    #                     reference_labels=target,
    #                     embeddings_come_from_same_source=True,
    #                 )

    #                 self.train_accuracy.update(metrics["precision_at_1"], image.size(0))
    #                 self.train_map_r.update(metrics["mean_average_precision_at_r"], image.size(0))

    #                 tqdm.write("=" * 60)
    #                 tqdm.write(
    #                     "Epoch {}/{} Batch (Step) {}/{}\t"
    #                     "Time {}\t"
    #                     "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
    #                     "Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
    #                     "Training Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
    #                     "Training MAP@r {map_r.val:.4f} ({map_r.avg:.4f})".format(
    #                         epoch + 1,
    #                         self.args.num_epoch,
    #                         batch + 1,
    #                         len(self.train_loader) * self.args.num_epoch,
    #                         time.asctime(time.localtime(time.time())),
    #                         loss=losses,
    #                         loss_KL=losses_KL,
    #                         acc=self.train_accuracy,
    #                         map_r=self.train_map_r,
    #                     )
    #                 )

    #             batch += 1

    #         self.writer.add_scalar(
    #             "train_loss", losses.avg, global_step=epoch, new_style=True
    #         )
    #         self.writer.add_scalar(
    #             "train_loss_KL", losses_KL.avg, global_step=epoch, new_style=True
    #         )
    #         self.writer.add_scalar(
    #             "train_accuracy", self.train_accuracy.avg, global_step=epoch, new_style=True
    #         )
    #         self.writer.add_scalar(
    #             "train_map_r", self.train_map_r.avg, global_step=epoch, new_style=True
    #         )

    #         # Validate @ frequency
    #         if (epoch + 1) % self.args.save_freq == 0:
    #             print("=" * 60, flush=True)
    #             self.validate()
    #             self.save_model()
    #             print("=" * 60, flush=True)

    #     print(f"Finished training @ epoch: {self.epoch + 1}")
    #     return self.model

    # def validate(self):
    #     print(f"Validating @ epoch: {self.epoch + 1}")

    #     self.model.eval()

    #     self.val_loss = AverageMeter()
    #     self.val_loss_KL = AverageMeter()
    #     self.val_accuracy = AverageMeter()
    #     self.val_map_r = AverageMeter()

    #     id_sigma = []
    #     id_mu = []
    #     with torch.no_grad():
    #         for image, target in tqdm(self.val_loader, desc="Validating"):
    #             mu_dul, std_dul = self.model(image)

    #             # Save for visualization
    #             id_sigma.append(std_dul)
    #             id_mu.append(mu_dul)

    #             # Reparameterization trick
    #             epsilon = torch.randn_like(std_dul)
    #             samples = mu_dul + epsilon * std_dul
    #             variance_dul = std_dul**2

    #             norm_samples = l2_norm(samples)

    #             hard_pairs = self.miner(norm_samples, target)
    #             loss = self.loss_fn(norm_samples, target, hard_pairs)

    #             loss_kl = (
    #                 ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
    #                 .sum(dim=-1)
    #                 .mean()
    #             )
    #             metrics = self.metrics.get_accuracy(
    #                 query=norm_samples,
    #                 reference=norm_samples,
    #                 query_labels=target,
    #                 reference_labels=target,
    #                 embeddings_come_from_same_source=True,
    #             )

    #             self.val_loss.update(loss.item(), image.size(0))
    #             self.val_loss_KL.update(loss_kl.data.item(), image.size(0))

    #             self.val_accuracy.update(metrics["precision_at_1"], image.size(0))
    #             self.val_map_r.update(metrics["mean_average_precision_at_r"], image.size(0))

    #     self.writer.add_scalar(
    #         "val_loss", self.val_loss.avg, global_step=self.epoch, new_style=True
    #     )
    #     self.writer.add_scalar(
    #         "val_loss_KL", self.val_loss_KL.avg, global_step=self.epoch, new_style=True
    #     )

    #     # accuracy = test_model(
    #     #     self.train_loader.dataset, self.val_loader.dataset, self.model, self.device,
    #     #     self.args.batch_size, self.args.num_workers
    #     # )

    #     self.writer.add_scalar(
    #         "val_accuracy",
    #         self.val_accuracy.avg,
    #         global_step=self.epoch,
    #         new_style=True,
    #     )
    #     self.writer.add_scalar(
    #         "val_map_r",
    #         self.val_map_r.avg,
    #         global_step=self.epoch,
    #         new_style=True,
    #     )

    #     # dispaly training loss & acc every DISP_FREQ
    #     print(
    #         "Time {}\t"
    #         "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
    #         "Validation Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
    #         "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
    #         "Validation MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
    #             time.asctime(time.localtime(time.time())),
    #             loss=self.val_loss,
    #             loss_KL=self.val_loss_KL,
    #             acc=self.val_accuracy,
    #             map_r=self.val_map_r,
    #         ),
    #         flush=True,
    #     )

    #     if self.to_visualize:
    #         print("=" * 60, flush=True)
    #         print("Visualizing...")

    #         ood_sigma = []
    #         ood_mu = []
    #         for img, label in tqdm(self.ood_loader, desc="OOD"):
    #             mu_dul, std_dul = self.model(img)
    #             ood_sigma.append(std_dul)
    #             ood_mu.append(mu_dul)

    #         id_sigma = torch.cat(id_sigma, dim=0).cpu().detach().numpy()
    #         id_mu = torch.cat(id_mu, dim=0).cpu().detach().numpy()
    #         ood_sigma = torch.cat(ood_sigma, dim=0).cpu().detach().numpy()
    #         ood_mu = torch.cat(ood_mu, dim=0).cpu().detach().numpy()

    #         self.visualize(
    #             id_mu, id_sigma, ood_mu, ood_sigma, prefix='val_'
    #         )

    #     self.model.train()

    # def test(self):
    #     print(f"Testing @ epoch: {self.epoch}")
    #     self.model.eval()

    #     self.test_accuracy = AverageMeter()
    #     self.test_map_r = AverageMeter()

    #     id_sigma = []
    #     id_mu = []
    #     with torch.no_grad():
    #         for image, target in tqdm(self.test_loader, desc="Testing"):
    #             mu_dul, std_dul = self.model(image)

    #             # Save for visualization
    #             id_sigma.append(std_dul)
    #             id_mu.append(mu_dul)

    #             # Reparameterization trick
    #             epsilon = torch.randn_like(std_dul)
    #             samples = mu_dul + epsilon * std_dul

    #             norm_samples = l2_norm(samples)

    #             metrics = self.metrics.get_accuracy(
    #                 query=norm_samples,
    #                 reference=norm_samples,
    #                 query_labels=target,
    #                 reference_labels=target,
    #                 embeddings_come_from_same_source=True,
    #             )

    #             self.test_accuracy.update(metrics["precision_at_1"], image.size(0))
    #             self.test_map_r.update(metrics["mean_average_precision_at_r"], image.size(0))

    #     self.writer.add_scalar(
    #         "test_accuracy",
    #         self.test_accuracy.avg,
    #         global_step=self.epoch,
    #         new_style=True,
    #     )
    #     self.writer.add_scalar(
    #         "test_map_r",
    #         self.test_map_r.avg,
    #         global_step=self.epoch,
    #         new_style=True,
    #     )

    #     print(
    #         "Time {}\t"
    #         "Testing Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
    #         "Testing MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
    #             time.asctime(time.localtime(time.time())),
    #             acc=self.test_accuracy,
    #             map_r=self.test_map_r,
    #         ),
    #         flush=True,
    #     )

    #     if self.to_visualize:
    #         print("=" * 60, flush=True)
    #         print("Visualizing...")

    #         ood_sigma = []
    #         ood_mu = []
    #         for img, _ in tqdm(self.ood_loader, desc="OOD"):
    #             mu_dul, std_dul = self.model(img)
    #             ood_sigma.append(std_dul)
    #             ood_mu.append(mu_dul)

    #         id_sigma = torch.cat(id_sigma, dim=0).cpu().detach().numpy()
    #         id_mu = torch.cat(id_mu, dim=0).cpu().detach().numpy()
    #         ood_sigma = torch.cat(ood_sigma, dim=0).cpu().detach().numpy()
    #         ood_mu = torch.cat(ood_mu, dim=0).cpu().detach().numpy()

    #         self.visualize(
    #             id_mu, id_sigma, ood_mu, ood_sigma, prefix='test_'
    #         )

    # def visualize(self, id_mu, id_sigma, ood_mu, ood_sigma, prefix):
    #     if not prefix.endswith('_'):
    #         prefix += '_'

    #     # Set path
    #     vis_path = self.args.vis_dir / self.name / f"epoch_{self.epoch + 1}"
    #     vis_path.mkdir(parents=True, exist_ok=True)

    #     # Visualize
    #     plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix)

    #     id_var = id_sigma**2
    #     ood_var = ood_sigma**2
    #     plot_ood(id_mu, id_var, ood_mu, ood_var, vis_path, prefix)

    # def forward(self, x):
    #     return self.model(x)

    # def log_hyperparams(self):
    #     print("Logging hyperparameters")

    #     hparams = vars(self.args)
    #     hparams['name'] = self.name
    #     hparams['epoch'] = self.epoch
    #     hparams['miner'] = self.miner.__class__.__name__
    #     hparams['model'] = self.model.module.module.__class__.__name__
    #     hparams['optimizer'] = self.optimizer.__class__.__name__
    #     hparams['loss_fn'] = self.loss_fn.__class__.__name__

    #     for key, val in hparams.items():
    #         if isinstance(val, PosixPath):
    #             hparams[key] = str(val)
    #         elif isinstance(val, list):
    #             hparams[key] = str(val)

    #     self.writer.add_hparams(
    #         hparam_dict=hparams,
    #         metric_dict={
    #             "train_accuracy": self.train_accuracy.avg,
    #             "train_map_r": self.train_map_r.avg,
    #             "val_accuracy": self.val_accuracy.avg,
    #             "val_map_r": self.val_map_r.avg,
    #             "test_accuracy": self.test_accuracy.avg,
    #             "test_map_r": self.test_map_r.avg,
    #         },
    #         run_name=".",
    #     )

    # def add_data_module(self, data_module):
    #     data_module.prepare_data()
    #     data_module.setup(shuffle=self.args.shuffle)

    #     self.train_loader, self.val_loader, self.test_loader, self.ood_loader = self.setup_dataloaders(
    #         data_module.train_dataloader(),
    #         data_module.val_dataloader(),
    #         data_module.test_dataloader(),
    #         data_module.ood_dataloader()
    #     )

    # def save_model(self, prefix = None):
    #     name = "Model_Epoch_{}_Time_{}_checkpoint.pth".format(
    #         self.epoch + 1, get_time()
    #     )

    #     if prefix is not None:
    #         name = prefix + "_" + name

    #     path = (
    #         Path(self.args.model_save_folder)
    #         / self.args.name
    #         / name
    #     )

    #     path.parent.mkdir(parents=True, exist_ok=True)

    #     print(f"Saving model @ {str(path)}")
    #     self.save(
    #         content=self.model.module.module.state_dict(), filepath=str(path)
    #     )
