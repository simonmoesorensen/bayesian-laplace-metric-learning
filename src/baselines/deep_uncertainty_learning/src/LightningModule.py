import datetime
import logging
from pathlib import Path
import time

import torch
from matplotlib import pyplot as plt
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN
import numpy as np
import pandas as pd
import torchmetrics
import json

from utils import AverageMeter, l2_norm, test_model

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class DULTrainer(LightningLite):
    def init(self, model, loss_fn, miner, optimizer, dul_args, to_visualize=False):
        print("Overall Configurations:")
        print("=" * 60)
        for k in dul_args.__dict__:
            print(" '{}' : '{}' ".format(k, str(dul_args.__dict__[k])))

        self.dul_args = dul_args

        # LOGGING
        self.name = dul_args.name
        self.writer = self.setup_logger(dul_args.name)

        self.to_visualize = to_visualize

        # Data
        self.batch_size = dul_args.batch_size

        # Load model
        if dul_args.model_path:
            state_dict = self.load(dul_args.model_path)
            model.load_state_dict(state_dict)

        # Miners and Loss
        self.loss_fn = loss_fn

        # REQUIRED FOR ARCFACE LOSS
        self.loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)

        self.miner = miner

        # Lite setup
        self.model, self.optimizer = self.setup(model, optimizer)

        # Required for ARCFACE loss
        knn_func = CustomKNN(CosineSimilarity())

        # Metric calculation
        self.metrics = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k='max_bin_count',
            device=self.device,
            knn_func=knn_func
        )

    def setup_logger(self, name):
        subdir = get_time()
        logdir = f"{self.dul_args.log_dir}/{name}/{subdir}"
        writer = SummaryWriter(logdir)
        return writer

    # noinspection PyMethodOverriding
    def run(self):

        print(f"Training")
        self.model.train()

        if not self.name:
            raise ValueError("Please run .init()")

        losses = AverageMeter()
        losses_KL = AverageMeter()
        accuracy = AverageMeter()
        map_r = AverageMeter()

        DISP_FREQ = len(self.train_loader) // 20  # frequency to display training loss
        batch = 0

        for epoch in range(self.dul_args.num_epoch):
            if epoch < self.dul_args.resume_epoch:
                continue

            self.epoch = epoch

            for image, target in tqdm(self.train_loader, desc='Training'):
                self.optimizer.zero_grad()

                mu_dul, std_dul = self.model(image)

                epsilon = torch.randn_like(std_dul)
                samples = mu_dul + epsilon * std_dul
                variance_dul = std_dul**2

                norm_samples = l2_norm(samples)

                hard_pairs = self.miner(norm_samples, target)
                loss_backbone = self.loss_fn(norm_samples, target, hard_pairs)

                loss_kl = (
                    ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
                    .sum(dim=-1)
                    .mean()
                )

                loss_backbone += self.dul_args.kl_scale * loss_kl

                self.backward(loss_backbone)
                self.optimizer.step()
                self.loss_optimizer.step()

                losses_KL.update(loss_kl.item(), image.size(0))
                losses.update(loss_backbone.data.item(), image.size(0))
                

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    # Metrics
                    metrics = self.metrics.get_accuracy(
                        query=norm_samples,
                        reference=norm_samples,
                        query_labels=target,
                        reference_labels=target,
                        embeddings_come_from_same_source=True,
                    )

                    accuracy.update(metrics["precision_at_1"], image.size(0))
                    map_r.update(metrics["mean_average_precision_at_r"], image.size(0))

                    tqdm.write("=" * 60)
                    tqdm.write(
                        "Epoch {}/{} Batch (Step) {}/{}\t"
                        "Time {}\t"
                        "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
                        "Training Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
                        "Training MAP@r {map_r.val:.4f} ({map_r.avg:.4f})".format(
                            epoch + 1,
                            self.dul_args.num_epoch,
                            batch + 1,
                            len(self.train_loader) * self.dul_args.num_epoch,
                            time.asctime(time.localtime(time.time())),
                            loss=losses,
                            loss_KL=losses_KL,
                            acc=accuracy,
                            map_r=map_r,
                        )
                    )

                batch += 1

            self.writer.add_scalar(
                "train_loss", losses.avg, global_step=epoch, new_style=True
            )
            self.writer.add_scalar(
                "train_loss_KL", losses_KL.avg, global_step=epoch, new_style=True
            )
            self.writer.add_scalar(
                "train_accuracy", accuracy.avg, global_step=epoch, new_style=True
            )
            self.writer.add_scalar(
                "train_map_r", map_r.avg, global_step=epoch, new_style=True
            )

            # Validate @ frequency
            if (epoch + 1) % self.dul_args.save_freq == 0:
                print("=" * 60, flush=True)
                self.validate()

                backbone_path = (
                    Path(self.dul_args.model_save_folder)
                    / self.dul_args.name
                    / "Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                        epoch + 1, batch, get_time()
                    )
                )

                backbone_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"Saving model @ {str(backbone_path)}")
                self.save(
                    content=self.model.module.state_dict(), filepath=str(backbone_path)
                )
                print("=" * 60, flush=True)

            losses.reset()
            losses_KL.reset()
            accuracy.reset()
            map_r.reset()

        print(f"Finished training @ epoch: {self.epoch + 1}")
        return self.model

    def validate(self):
        print(f"Validating @ epoch: {self.epoch + 1}")

        self.model.eval()

        val_loss = AverageMeter()
        val_loss_KL = AverageMeter()
        val_accuracy = AverageMeter()
        val_map_r = AverageMeter()

        id_sigma = []
        with torch.no_grad():
            for image, target in tqdm(self.val_loader, desc="Validating"):
                mu_dul, std_dul = self.model(image)

                # Save for visualization
                id_sigma.append(std_dul)

                # Reparameterization trick
                epsilon = torch.randn_like(std_dul)
                samples = mu_dul + epsilon * std_dul
                variance_dul = std_dul**2

                norm_samples = l2_norm(samples)

                hard_pairs = self.miner(norm_samples, target)
                loss = self.loss_fn(norm_samples, target, hard_pairs)

                loss_kl = (
                    ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
                    .sum(dim=-1)
                    .mean()
                )
                metrics = self.metrics.get_accuracy(
                    query=norm_samples,
                    reference=norm_samples,
                    query_labels=target,
                    reference_labels=target,
                    embeddings_come_from_same_source=True,
                )

                val_loss.update(loss.item(), image.size(0))
                val_loss_KL.update(loss_kl.data.item(), image.size(0))

                val_accuracy.update(metrics["precision_at_1"], image.size(0))
                val_map_r.update(metrics["mean_average_precision_at_r"], image.size(0))

        self.writer.add_scalar(
            "val_loss", val_loss.avg, global_step=self.epoch, new_style=True
        )
        self.writer.add_scalar(
            "val_loss_KL", val_loss_KL.avg, global_step=self.epoch, new_style=True
        )

        # accuracy = test_model(
        #     self.train_loader.dataset, self.val_loader.dataset, self.model, self.device,
        #     self.dul_args.batch_size, self.dul_args.num_workers
        # )

        self.writer.add_scalar(
            "val_accuracy",
            val_accuracy.avg,
            global_step=self.epoch,
            new_style=True,
        )
        self.writer.add_scalar(
            "val_map_r",
            val_map_r.avg,
            global_step=self.epoch,
            new_style=True,
        )

        # dispaly training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
            "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Validation MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
                time.asctime(time.localtime(time.time())),
                loss=val_loss,
                loss_KL=val_loss_KL,
                acc=val_accuracy,
                map_r=val_map_r,
            ),
            flush=True,
        )

        if self.to_visualize:
            print("=" * 60, flush=True)
            print("Visualizing...")

            ood_sigma = []
            for img, label in tqdm(self.ood_loader, desc="OOD"):
                mu_dul, std_dul = self.model(img)
                ood_sigma.append(std_dul)

            id_sigma = torch.cat(id_sigma, dim=0).cpu().detach().numpy()
            ood_sigma = torch.cat(ood_sigma, dim=0).cpu().detach().numpy()

            self.visualize(
                id_sigma, ood_sigma, prefix='val_'
            )

        self.model.train()

    def test(self):
        print(f"Testing @ epoch: {self.epoch}")
        accuracy = test_model(
            self.train_loader.dataset,
            self.test_loader.dataset,
            self.model,
            self.device,
            self.dul_args.batch_size,
            self.dul_args.num_workers,
        )

        self.writer.add_scalar("test_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar(
            "test_map", accuracy["mean_average_precision"], self.epoch
        )

        if self.to_visualize:
            self.visualize(self.test_loader, self.test_loader.dataset.class_to_idx)

    def visualize(self, id_sigma, ood_sigma, prefix):
        if not prefix.endswith('_'):
            prefix += '_'

        # Set path
        vis_path = self.dul_args.vis_dir / self.name / f"epoch_{self.epoch + 1}"
        vis_path.mkdir(parents=True, exist_ok=True)

        id_sigma = np.reshape(id_sigma, (id_sigma.shape[0], -1))
        ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

        id_sigma, ood_sigma = id_sigma.sum(axis=1), ood_sigma.sum(axis=1)

        pred = np.concatenate([id_sigma, ood_sigma])
        target = np.concatenate([[0] * len(id_sigma), [1] * len(ood_sigma)])

        # plot roc curve
        roc = torchmetrics.ROC(num_classes=1)
        fpr, tpr, thresholds = roc(
            torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
        )

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title('OOD ROC Curve')
        plt.legend()
        fig.savefig(vis_path / f"{prefix}ood_roc_curve.png")
        plt.cla()
        plt.close()

        # save data
        data = pd.DataFrame(
            np.concatenate([pred[:, None], target[:, None]], axis=1),
            columns=["sigma", "labels"],
        )
        data.to_csv(vis_path / f"{prefix}ood_roc_curve_data.csv")

        # plot precision recall curve
        pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
        precision, recall, thresholds = pr_curve(
            torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
        )

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title('OOD Precision-Recall Curve')
        plt.legend()
        fig.savefig(vis_path / f"{prefix}ood_precision_recall_curve.png")
        plt.cla()
        plt.close()

        metrics = {}

        # compute auprc (area under precission recall curve)
        auc = torchmetrics.AUC(reorder=True)
        auprc_score = auc(recall, precision)
        metrics["auprc"] = float(auprc_score.numpy())

        # compute false positive rate at 80
        num_id = len(id_sigma)

        for p in range(0, 100, 10):
            # if there is no difference in variance
            try:
                metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
            except:
                metrics[f"fpr{p}"] = "none"
            else:
                continue

        # compute auroc
        auroc = torchmetrics.AUROC(num_classes=1)
        auroc_score = auroc(
            torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
        )
        metrics["auroc"] = float(auroc_score.numpy())
        print(f"Metrics: {metrics}")
        
        # save metrics
        with open(vis_path / "ood_metrics.json", "w") as outfile:
            json.dump(metrics, outfile)


    def forward(self, x):
        return self.model(x)

    def log_hyperparams(self):
        print("Logging hyperparameters")

        train_accuracy = test_model(
            self.train_loader.dataset,
            self.train_loader.dataset,
            self.model,
            self.device,
            self.dul_args.batch_size,
            self.dul_args.num_workers,
        )
        print(f"{train_accuracy=}")

        val_accuracy = test_model(
            self.train_loader.dataset,
            self.val_loader.dataset,
            self.model,
            self.device,
            self.dul_args.batch_size,
            self.dul_args.num_workers,
        )
        print(f"{val_accuracy=}")

        print("Calculating test accuracy")
        test_accuracy = test_model(
            self.train_loader.dataset,
            self.test_loader.dataset,
            self.model,
            self.device,
            self.dul_args.batch_size,
            self.dul_args.num_workers,
        )
        print(f"{test_accuracy=}")

        self.writer.add_hparams(
            hparam_dict={
                "name": self.name,
                "miner": self.miner.__class__.__name__,
                "loss_fn": self.loss_fn.__class__.__name__,
                "epoch": self.epoch + 1,
                "lr": self.optimizer.defaults["lr"],
                "batch_size": self.batch_size,
                "model": self.model.module.__class__.__name__,
            },
            metric_dict={
                "train_acc": train_accuracy["precision_at_1"],
                "train_map": train_accuracy["mean_average_precision"],
                "val_acc": val_accuracy["precision_at_1"],
                "val_map": val_accuracy["mean_average_precision"],
                "test_acc": test_accuracy["precision_at_1"],
                "test_map": test_accuracy["mean_average_precision"],
            },
            run_name=".",
        )

    def add_data_module(self, data_module):
        data_module.prepare_data()
        data_module.setup(shuffle=self.dul_args.shuffle)

        self.train_loader, self.val_loader, self.test_loader, self.ood_loader = self.setup_dataloaders(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
            data_module.ood_dataloader()
        )
