import datetime
import json
import logging
import time
from pathlib import Path, PosixPath

import torch
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt
from pytorch_lightning.lite import LightningLite
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN
from src.distances import ExpectedSquareL2Distance
from src.metrics.MetricMeter import AverageMeter, MetricMeter
from src.recall_at_k import AccuracyRecall
from src.visualize import (
    get_names,
    visualize_all,
    plot_calibration_curve,
    plot_sparsification_curve,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.evaluate import compute_map_k, compute_recall_k, compute_rank, compute_pidx

from src.utils import filter_state_dict, get_pairs

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class BaseLightningModule(LightningLite):
    """
    Base Lightning Module for Probabilistic Metric Learning
    """

    def init(self, model, loss_fn, miner, optimizer, args):
        print("Overall Configurations:")
        print("=" * 60)
        for k in args.__dict__:
            print(" '{}' : '{}' ".format(k, str(args.__dict__[k])))

        torch.manual_seed(args.random_seed)

        self.args = args

        # LOGGING
        self.name = args.name
        self.writer = self.setup_logger(args.name)

        self.to_visualize = args.to_visualize

        # Load model
        if args.model_path:
            state_dict = self.load(args.model_path)

            model.load_state_dict(filter_state_dict(state_dict))

        # Data
        self.batch_size = args.batch_size

        # Miners and Loss
        self.loss_fn = self.to_device(loss_fn)

        self.miner = miner

        # Lite setup
        self.model, self.optimizer = self.setup(model, optimizer)

        knn_func = CustomKNN(distances.LpDistance())

        # Metric calculation
        self.metric_calc = AccuracyRecall(
            include=("mean_average_precision", "precision_at_1", "recall_at_k"),
            k=5,
            device=self.device,
            knn_func=knn_func,
        )

        knn_func_expected = CustomKNN(ExpectedSquareL2Distance())

        self.metric_calc_expected = AccuracyRecall(
            include=("mean_average_precision", "precision_at_1", "recall_at_k"),
            k=5,
            device=self.device,
            knn_func=knn_func_expected,
        )

        # Meters
        self.metrics = MetricMeter(
            meters={
                "train/accuracy": AverageMeter(),
                "train/map_k": AverageMeter(),
                "train/recall_k": AverageMeter(),
                "val/accuracy": AverageMeter(),
                "val/map_k": AverageMeter(),
                "val/recall_k": AverageMeter(),
                "test/accuracy": AverageMeter(),
                "test/map_k": AverageMeter(),
                "test/recall_k": AverageMeter(),
                "train/loss": AverageMeter(),
                "val/loss": AverageMeter(),
                "hessian/norm": AverageMeter(),
                "hessian/min": AverageMeter(),
                "hessian/max": AverageMeter(),
                "hessian/avg": AverageMeter(),
                "curr_hessian/norm": AverageMeter(),
                "curr_hessian/min": AverageMeter(),
                "curr_hessian/max": AverageMeter(),
                "curr_hessian/avg": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

        self.expected_metrics = MetricMeter(
            meters={
                "train_expected/accuracy": AverageMeter(),
                "train_expected/map_k": AverageMeter(),
                "train_expected/recall_k": AverageMeter(),
                "val_expected/accuracy": AverageMeter(),
                "val_expected/map_k": AverageMeter(),
                "val_expected/recall_k": AverageMeter(),
                "test_expected/accuracy": AverageMeter(),
                "test_expected/map_k": AverageMeter(),
                "test_expected/recall_k": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

        self.additional_metrics = MetricMeter(
            meters={
                "val/ece": AverageMeter(),
                "val/ausc": AverageMeter(),
                "val/auroc": AverageMeter(),
                "val/auprc": AverageMeter(),
                "test/ece": AverageMeter(),
                "test/ausc": AverageMeter(),
                "test/auroc": AverageMeter(),
                "test/auprc": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

        self.epoch = 0

    def setup_logger(self, name):
        subdir = get_time()
        logdir = Path(self.args.log_dir) / self.args.dataset / name / subdir
        writer = SummaryWriter(logdir)
        return writer

    def log(self, metrics):
        for metric in metrics:
            self.writer.add_scalar(
                f"{metric}",
                self.metrics.get(metric).avg,
                global_step=self.epoch + 1,
                new_style=True,
            )

    def log_additional(self, metrics):
        for metric in metrics:
            self.writer.add_scalar(
                f"{metric}",
                self.additional_metrics.get(metric).avg,
                global_step=self.epoch + 1,
                new_style=True,
            )

    def train_step(self, X, y):
        raise NotImplementedError()

    def val_step(self, X, y):
        raise NotImplementedError()

    def test_step(self, X, y):
        raise NotImplementedError()

    def ood_step(self, X, y):
        raise NotImplementedError()

    def epoch_start(self):
        self.metrics.reset(
            ["train/loss", "train/accuracy", "train/map_k", "train/recall_k"]
        )

    def epoch_end(self):
        self.log(["train/loss", "train/accuracy", "train/map_k", "train/recall_k"])

    def train_start(self):
        pass

    def train_end(self):
        pass

    def val_start(self):
        self.metrics.reset(["val/loss", "val/accuracy", "val/map_k", "val/recall_k"])

    def val_end(self):
        self.log(["val/loss", "val/accuracy", "val/map_k", "val/recall_k"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Validation MAP@k {map_k.val:.4f} ({map_k.avg:.4f})\t"
            "Validation Recall@k {recall_k.val:4f} ({recall_k.avg:.4f})".format(
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("val/loss"),
                acc=self.metrics.get("val/accuracy"),
                map_k=self.metrics.get("val/map_k"),
                recall_k=self.metrics.get("val/recall_k"),
            ),
            flush=True,
        )

    def test_start(self):
        self.metrics.reset(["test/accuracy", "test/map_k", "test/recall_k"])

    def test_end(self):
        self.log(["test/accuracy", "test/map_k", "test/recall_k"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Test Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Test MAP@k {map_k.val:.4f} ({map_k.avg:.4f})\t"
            "Test Recall@k {recall_k.val:.4f} ({recall_k.avg:.4f})".format(
                time.asctime(time.localtime(time.time())),
                acc=self.metrics.get("test/accuracy"),
                map_k=self.metrics.get("test/map_k"),
                recall_k=self.metrics.get("test/recall_k"),
            ),
            flush=True,
        )

    def display(self, epoch, batch):
        tqdm.write("=" * 60)
        tqdm.write(
            "Epoch {}/{} Batch (Step) {}/{}\t"
            "Time {}\t"
            "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Training Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Training MAP@k {map_k.val:.4f} ({map_k.avg:.4f})\t"
            "Training Recall@k {recall_k.val:.4f} ({recall_k.avg:.4f})\t"
            "Lr {lr:.4f}".format(
                epoch + 1,
                self.args.num_epoch,
                batch + 1,
                len(self.train_loader) * self.args.num_epoch,
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("train/loss"),
                acc=self.metrics.get("train/accuracy"),
                map_k=self.metrics.get("train/map_k"),
                recall_k=self.metrics.get("train/recall_k"),
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

    def update_accuracy(
        self, z, y, step="train", z_db=None, y_db=None, same_source=False
    ):
        if step not in ["train", "val", "test"]:
            raise ValueError("step must be one of ['train', 'val', 'test']")

        if z_db is None:
            z_db = z

        if y_db is None:
            y_db = y

        # Metrics
        with torch.no_grad():
            metrics = self.metric_calc.get_accuracy(
                query=z,
                reference=z_db,
                query_labels=y,
                reference_labels=y_db,
                embeddings_come_from_same_source=same_source,
            )

            self.metrics.update(f"{step}/accuracy", metrics["precision_at_1"])
            self.metrics.update(f"{step}/map_k", metrics["mean_average_precision"])
            self.metrics.update(f"{step}/recall_k", metrics["recall_at_k"])

    def update_expected_accuracy(
        self, z, y, step="train", z_db=None, y_db=None, same_source=False
    ):
        if step not in ["train", "val", "test"]:
            raise ValueError("step must be one of ['train', 'val', 'test']")

        if z_db is None:
            z_db = z

        if y_db is None:
            y_db = y

        # Metrics
        with torch.no_grad():
            metrics = self.metric_calc_expected.get_accuracy(
                query=z,
                reference=z_db,
                query_labels=y,
                reference_labels=y_db,
                embeddings_come_from_same_source=same_source,
            )

            self.expected_metrics.update(
                f"{step}_expected/accuracy", metrics["precision_at_1"]
            )
            self.expected_metrics.update(
                f"{step}_expected/map_k", metrics["mean_average_precision"]
            )
            self.expected_metrics.update(
                f"{step}_expected/recall_k", metrics["recall_at_k"]
            )

    def optimizer_step(self):
        self.optimizer.step()

    def run(self):
        self.train()

    # noinspection PyMethodOverriding
    def train(self):
        print("Training")
        self.train_start()
        self.model.train()

        if not self.name:
            raise ValueError("Please run .init()")

        DISP_FREQ = (
            len(self.train_loader) // self.args.disp_freq
        )  # frequency to display training loss
        batch = 0

        for epoch in range(self.args.num_epoch):
            self.epoch_start()

            if epoch < self.args.resume_epoch:
                continue

            self.epoch = epoch

            for image, target, class_labels in tqdm(self.train_loader, desc="Training"):
                self.optimizer.zero_grad()
                
                bs, nobs, c, h, w = image.shape
                image = image.view(bs * nobs, c, h, w)
                
                pairs = get_pairs(target)

                out, loss = self.train_step(image, pairs)

                self.backward(loss)

                self.optimizer_step()

                # display and log metrics every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    with torch.no_grad():
                        
                        class_labels = class_labels.view(-1)
                        self.update_accuracy(
                            out,
                            class_labels,
                            step="train",
                            same_source=True,
                        )
                        self.display(epoch, batch)
                        self.writer.add_scalar(
                            "lr",
                            self.optimizer.param_groups[0]["lr"],
                            global_step=self.epoch + 1,
                            new_style=True,
                        )

                batch += 1

            self.epoch_end()

            # Validate @ frequency
            if (epoch + 1) % self.args.save_freq == 0:
                print("=" * 60, flush=True)
                self.validate()
                self.save_model()
                print("=" * 60, flush=True)

        self.train_end()
        print(f"Finished training @ epoch: {self.epoch + 1}")
        return self.model

    def validate(self):
        print(f"Validating @ epoch: {self.epoch + 1}")

        self.model.eval()

        self.val_start()

        val_sigma = []
        val_mu = []
        val_images = []
        val_labels = []
        val_samples = []
        with torch.no_grad():
            for image, target in tqdm(self.val_loader, desc="Validating"):
                mu, sigma, samples = self.val_step(image, target)
                
                if sigma is not None:
                    val_sigma.append(sigma)
                val_mu.append(mu)
                val_images.append(image)
                val_labels.append(target)
                val_samples.append(samples)

                self.update_accuracy(
                    mu,
                    target,
                    step="val",
                    same_source=True,
                )

                if sigma is not None:
                    self.update_expected_accuracy(
                        torch.stack((mu, sigma**2), dim=-1),
                        target,
                        step="val",
                        same_source=True,
                    )

        self.val_end()

        if self.to_visualize:
            hessian = self.hessian if hasattr(self, "hessian") else None
            self.visualize(val_mu, val_sigma, val_images, val_labels, val_samples, hessian, prefix="val")

        self.model.train()

    def compute_features(self, loader):
        self.model.eval()

        z_mu = []
        z_sigma = []
        z_samples = []
        labels = []
        with torch.no_grad():
            for image, target in tqdm(loader, desc="Computing features"):
                mu, sigma, samples = self.test_step(image, target)
                z_mu.append(mu.cpu())
                z_sigma.append(sigma.cpu())
                z_samples.append(samples.cpu())
                labels.append(target)

        z_mu = torch.cat(z_mu, dim=0)
        z_sigma = torch.cat(z_sigma, dim=0)
        z_samples = torch.cat(z_samples, dim=1).permute(1, 0, 2)
        labels = torch.cat(labels, dim=0)

        return z_mu, z_sigma, z_samples, labels
    
    def test(self, expected=True):
        print(f"Testing @ epoch: {self.epoch + 1}")
        self.model.eval()

        self.test_start()
        
        z_mu, z_sigma, z_samples, labels = self.compute_features(self.test_loader)
        # ood_z_mu, ood_z_sigma, ood_z_samples, ood_labels = self.compute_features(self.ood_loader)
        
        ks = [1, 5, 10, 20]
        
        pos_idx = compute_pidx(labels.cpu().numpy())
        rank = compute_rank(z_mu.numpy(), None, samesource=True)
        mapk = [compute_map_k(rank, pos_idx, k) for k in ks]
        recallk = compute_recall_k(rank, pos_idx, ks)
        
        if expected:
            rank = compute_rank(z_mu, z_sigma, samesource=True)
            expected_mapk = [compute_map_k(rank, pos_idx, k) for k in ks]
            expected_recallk = compute_recall_k(rank, pos_idx, ks)
            
        self.test_end()
        
        if self.to_visualize:
            hessian = self.hessian if hasattr(self, "hessian") else None
            self.visualize(
                test_mu, test_sigma, test_images, test_labels, test_samples, hessian, prefix="test"
            )

    def visualize(self, id_mu, id_sigma, id_images, id_labels, id_samples, hessian, prefix):
        
        prob_model = (len(id_sigma) > 0) and (id_sigma is not None)
                
        if prob_model:
            id_sigma = torch.cat(id_sigma, dim=0).cpu()
        id_mu = torch.cat(id_mu, dim=0).cpu()
        id_images = torch.cat(id_images, dim=0).cpu()
        id_labels = torch.cat(id_labels, dim=0).cpu()
        
        # N, num_samples, D
        id_samples = torch.cat(id_samples, dim=1).cpu().permute(1,0,2)

        print("=" * 60, flush=True)
        print("Visualizing...")

        # Set path
        vis_path = (
            Path(self.args.vis_dir)
            / self.args.dataset
            / self.name
            / f"epoch_{self.epoch + 1}"
        )
        vis_path.mkdir(parents=True, exist_ok=True)
        
        if hessian is not None:
            plt.plot(hessian.cpu().numpy())
            plt.yscale("log")
            plt.savefig(vis_path / "hessian.png")

        if prob_model:
            ood_sigma = []
            ood_mu = []
            ood_images = []
            ood_labels = []

            with torch.no_grad():
                for img, y in tqdm(self.ood_loader, desc="OOD"):
                    out = self.ood_step(img, y)
                    if len(out) == 2:
                        mu_ood, std_ood = out
                    elif len(out) == 3:
                        mu_ood, std_ood, _ = out
                    else:
                        raise ValueError("Invalid output from OOD step")

                    ood_sigma.append(std_ood)
                    ood_mu.append(mu_ood)
                    ood_images.append(img)
                    ood_labels.append(y)

            ood_sigma = torch.cat(ood_sigma, dim=0).cpu()
            ood_mu = torch.cat(ood_mu, dim=0).cpu()
            ood_images = torch.cat(ood_images, dim=0).cpu()
            ood_labels = torch.cat(ood_labels, dim=0).cpu()

            # Visualize
            visualize_all(
                id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
            )

            model_name, dataset_name, run_name = get_names(vis_path)

            print("Running calibration curve")
            ece = plot_calibration_curve(
                id_labels,
                id_samples,
                vis_path,
                model_name,
                dataset_name,
                run_name,
            )
            self.additional_metrics.update(f"{prefix}/ece", ece)

            print("Running sparsification curve")

            ausc = plot_sparsification_curve(
                id_labels, id_mu, id_sigma, vis_path, model_name, dataset_name, run_name
            )

            self.additional_metrics.update(f"{prefix}/ausc", ausc)

            # Read ood metrics
            with open(vis_path / "ood_metrics.json", "r") as f:
                ood_metrics = json.load(f)
                self.additional_metrics.update(f"{prefix}/auroc", ood_metrics["auroc"])
                self.additional_metrics.update(f"{prefix}/auprc", ood_metrics["auprc"])

        # Save metrics
        metrics = self.metrics.get_dict()
        with open(vis_path / "metrics.json", "w") as f:
            json.dump(metrics, f)

        if prob_model:
            expected_metrics = self.expected_metrics.get_dict()
            with open(vis_path / "expected_metrics.json", "w") as f:
                json.dump(expected_metrics, f)

            additional_metrics = self.additional_metrics.get_dict()
            with open(vis_path / "additional_metrics.json", "w") as f:
                json.dump(additional_metrics, f)

        # Save hparams
        with open(vis_path / "hparams.json", "w") as f:
            json.dump(self.get_hparams(), f)

        if prob_model:
            # Save additional metrics for tensorboard in log_hyperparams
            add_metrics = [
                f"{prefix}/ece",
                f"{prefix}/ausc",
                f"{prefix}/auroc",
                f"{prefix}/auprc",
            ]

            # Update tensorboard
            self.log_additional(add_metrics)

    def forward(self, x):
        return self.model(x)

    def get_hparams(self):
        hparams = vars(self.args)
        hparams["name"] = self.name
        hparams["epoch"] = self.epoch
        hparams["miner"] = self.miner.__class__.__name__
        hparams["model"] = self.model.module.__class__.__name__
        hparams["optimizer"] = self.optimizer.__class__.__name__
        hparams["loss_fn"] = self.loss_fn.__class__.__name__

        for key, val in hparams.items():
            if isinstance(val, PosixPath):
                hparams[key] = str(val)
            elif isinstance(val, list):
                hparams[key] = str(val)

        return hparams

    def log_hyperparams(self):
        print("Logging hyperparameters")
        metrics = self.metrics.get_dict()
        additional_metrics = self.additional_metrics.get_dict()

        # Join metrics
        metrics.update(additional_metrics)

        hparams = self.get_hparams()
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict=metrics,
            run_name=".",
        )

    def add_data_module(self, data_module):

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.ood_loader,
        ) = self.setup_dataloaders(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
            data_module.ood_dataloader(),
        )

    def save_model(self, prefix=None):
        name = "Model_Epoch_{}_Time_{}_checkpoint.pth".format(
            self.epoch + 1, get_time()
        )

        if prefix is not None:
            name = prefix + "_" + name

        path = (
            Path(self.args.model_save_folder)
            / self.args.dataset
            / self.args.name
            / name
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model @ {str(path)}")
        self.save(content=self.model.module.state_dict(), filepath=str(path))
