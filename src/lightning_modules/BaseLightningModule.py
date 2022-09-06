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

        # Learning rate scheduler options
        self.base_lr = args.lr
        max_lr = args.lr * 10
        # Cycle every 5% of total epochs, results in base_lr around 60% of total epochs
        # See https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling?scriptVersionId=38549725&cellId=17
        step_size_up = max(1, args.num_epoch // 20)

        print(" 'base_lr' : '{}' ".format(self.base_lr))
        print(" 'max_lr' : '{}' ".format(max_lr))
        print(" 'step_size_up' : '{}' ".format(step_size_up))

        self.args = args

        # LOGGING
        self.name = args.name
        self.writer = self.setup_logger(args.name)

        self.to_visualize = args.to_visualize

        # Load model
        if args.model_path:
            state_dict = self.load(args.model_path)

            new_state_dict = {}
            for key in state_dict:
                if key.startswith("module."):
                    new_state_dict[key[7:]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]

            model.load_state_dict(new_state_dict)

        # Data
        self.batch_size = args.batch_size

        # Miners and Loss
        self.loss_fn = self.to_device(loss_fn)

        self.miner = miner

        # Lite setup
        self.model, self.optimizer = self.setup(model, optimizer)

        # Scheduler
        self.scheduler = lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode="triangular2",
            cycle_momentum=False,
        )

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
                "train_accuracy": AverageMeter(),
                "train_map_k": AverageMeter(),
                "train_recall_k": AverageMeter(),
                "val_accuracy": AverageMeter(),
                "val_map_k": AverageMeter(),
                "val_recall_k": AverageMeter(),
                "test_accuracy": AverageMeter(),
                "test_map_k": AverageMeter(),
                "test_recall_k": AverageMeter(),
                "train_loss": AverageMeter(),
                "val_loss": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

        self.expected_metrics = MetricMeter(
            meters={
                "train_expected_accuracy": AverageMeter(),
                "train_expected_map_k": AverageMeter(),
                "train_expected_recall_k": AverageMeter(),
                "val_expected_accuracy": AverageMeter(),
                "val_expected_map_k": AverageMeter(),
                "val_expected_recall_k": AverageMeter(),
                "test_expected_accuracy": AverageMeter(),
                "test_expected_map_k": AverageMeter(),
                "test_expected_recall_k": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

        self.additional_metrics = MetricMeter(
            meters={
                "val_ece": AverageMeter(),
                "val_ausc": AverageMeter(),
                "val_auroc": AverageMeter(),
                "val_auprc": AverageMeter(),
                "test_ece": AverageMeter(),
                "test_ausc": AverageMeter(),
                "test_auroc": AverageMeter(),
                "test_auprc": AverageMeter(),
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
            ["train_loss", "train_accuracy", "train_map_k", "train_recall_k"]
        )

    def epoch_end(self):
        self.log(["train_loss", "train_accuracy", "train_map_k", "train_recall_k"])

    def train_start(self):
        pass

    def train_end(self):
        pass

    def val_start(self):
        self.metrics.reset(["val_loss", "val_accuracy", "val_map_k", "val_recall_k"])

    def val_end(self):
        self.log(["val_loss", "val_accuracy", "val_map_k", "val_recall_k"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Validation MAP@k {map_k.val:.4f} ({map_k.avg:.4f})\t"
            "Validation Recall@k {recall_k.val:4f} ({recall_k.avg:.4f})".format(
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("val_loss"),
                acc=self.metrics.get("val_accuracy"),
                map_k=self.metrics.get("val_map_k"),
                recall_k=self.metrics.get("val_recall_k"),
            ),
            flush=True,
        )

    def test_start(self):
        self.metrics.reset(["test_accuracy", "test_map_k", "test_recall_k"])

    def test_end(self):
        self.log(["test_accuracy", "test_map_k", "test_recall_k"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Test Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Test MAP@k {map_k.val:.4f} ({map_k.avg:.4f})\t"
            "Test Recall@k {recall_k.val:.4f} ({recall_k.avg:.4f})".format(
                time.asctime(time.localtime(time.time())),
                acc=self.metrics.get("test_accuracy"),
                map_k=self.metrics.get("test_map_k"),
                recall_k=self.metrics.get("test_recall_k"),
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
                loss=self.metrics.get("train_loss"),
                acc=self.metrics.get("train_accuracy"),
                map_k=self.metrics.get("train_map_k"),
                recall_k=self.metrics.get("train_recall_k"),
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

            self.metrics.update(f"{step}_accuracy", metrics["precision_at_1"])
            self.metrics.update(f"{step}_map_k", metrics["mean_average_precision"])
            self.metrics.update(f"{step}_recall_k", metrics["recall_at_k"])

    def update_expected_accuracy(self, z, y, step="train", z_db=None, y_db=None):
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
                embeddings_come_from_same_source=False,
            )

            self.expected_metrics.update(
                f"{step}_expected_accuracy", metrics["precision_at_1"]
            )
            self.expected_metrics.update(
                f"{step}_expected_map_k", metrics["mean_average_precision"]
            )
            self.expected_metrics.update(
                f"{step}_expected_recall_k", metrics["recall_at_k"]
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

        # Save train data for accuracy calculation
        self.train_images = []
        self.train_labels = []

        for epoch in range(self.args.num_epoch):
            self.epoch_start()

            if epoch < self.args.resume_epoch:
                continue

            self.epoch = epoch

            for image, target in tqdm(self.train_loader, desc="Training"):
                if len(self.train_images) < len(self.train_loader.dataset):
                    self.train_images.append(image)
                    self.train_labels.append(target)

                self.optimizer.zero_grad()

                out, loss = self.train_step(image, target)

                self.backward(loss)

                self.optimizer_step()

                # display and log metrics every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    with torch.no_grad():
                        train_labels = torch.cat(self.train_labels, dim=0)

                        train_mu, _, _ = self.val_step(
                            torch.cat(self.train_images, dim=0), train_labels
                        )

                        self.update_accuracy(
                            out,
                            target,
                            step="train",
                            z_db=train_mu,
                            y_db=train_labels,
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
            self.scheduler.step()

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
        with torch.no_grad():
            if self.train_images:
                train_labels = torch.cat(self.train_labels, dim=0)

                train_mu, _, _ = self.val_step(
                    torch.cat(self.train_images, dim=0), train_labels
                )

            for image, target in tqdm(self.val_loader, desc="Validating"):
                mu, sigma, out = self.val_step(image, target)
                val_sigma.append(sigma)
                val_mu.append(mu)
                val_images.append(image)
                val_labels.append(target)

                self.update_accuracy(
                    out,
                    target,
                    step="val",
                    z_db=train_mu,
                    y_db=train_labels,
                )

                self.update_expected_accuracy(
                    torch.stack((val_mu, val_sigma), dim=-1),
                    target,
                    step="val",
                    z_db=train_mu,
                    y_db=train_labels,
                )

        self.val_end()

        if self.to_visualize:
            self.visualize(val_mu, val_sigma, val_images, val_labels, prefix="val_")

        self.model.train()

    def test(self, expected=True):
        print(f"Testing @ epoch: {self.epoch + 1}")
        self.model.eval()

        self.test_start()

        test_sigma = []
        test_mu = []
        test_images = []
        test_labels = []

        train_mu = []
        train_sigma = []
        train_sampled = []
        train_labels = []

        with torch.no_grad():
            for image, target in tqdm(
                self.train_loader, desc="Preparing query DB for testing"
            ):
                mu, sigma, out = self.test_step(image, target)
                train_mu.append(mu)
                train_sigma.append(sigma)
                train_sampled.append(out)
                train_labels.append(target)

            for image, target in tqdm(self.test_loader, desc="Testing"):
                mu, sigma, out = self.test_step(image, target)
                test_sigma.append(sigma)
                test_mu.append(mu)
                test_images.append(image)
                test_labels.append(target)

                self.update_accuracy(
                    mu,
                    target,
                    "test",
                    z_db=torch.cat(train_mu, dim=0),
                    y_db=torch.cat(train_labels, dim=0),
                )

                if expected:
                    self.update_expected_accuracy(
                        z=torch.stack((mu, sigma.square()), dim=-1),
                        y=target,
                        step="test",
                        z_db=torch.stack(
                            (
                                torch.cat(train_mu, dim=0),
                                torch.cat(train_sigma, dim=0).square(),
                            ),
                            dim=-1,
                        ),
                        y_db=torch.cat(train_labels, dim=0),
                    )

        self.test_end()

        if self.to_visualize:
            self.visualize(test_mu, test_sigma, test_images, test_labels, prefix="test_")

    def visualize(self, id_mu, id_sigma, id_images, id_labels, prefix):
        id_sigma = torch.cat(id_sigma, dim=0).detach().cpu()
        id_mu = torch.cat(id_mu, dim=0).detach().cpu()
        id_images = torch.cat(id_images, dim=0).detach().cpu()
        id_labels = torch.cat(id_labels, dim=0).detach().cpu()

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

        ood_sigma = []
        ood_mu = []
        ood_images = []
        ood_labels = []

        with torch.no_grad():
            for img, y in tqdm(self.ood_loader, desc="OOD"):
                mu_ood, std_ood, samples_ood = self.ood_step(img, y)
                ood_sigma.append(std_ood)
                ood_mu.append(mu_ood)
                ood_images.append(img)
                ood_labels.append(y)

        ood_sigma = torch.cat(ood_sigma, dim=0).detach().cpu()
        ood_mu = torch.cat(ood_mu, dim=0).detach().cpu()
        ood_images = torch.cat(ood_images, dim=0).detach().cpu()
        ood_labels = torch.cat(ood_labels, dim=0).detach().cpu()

        # Visualize
        visualize_all(
            id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
        )

        model_name, dataset_name, run_name = get_names(vis_path)

        print("Running calibration curve")
        ece = plot_calibration_curve(
            id_labels,
            id_mu,
            id_sigma,
            100,
            vis_path,
            model_name,
            dataset_name,
            run_name,
        )
        self.additional_metrics.update(f"{prefix}ece", ece)

        print("Running sparsification curve")

        ausc = plot_sparsification_curve(
            id_labels, id_mu, id_sigma, vis_path, model_name, dataset_name, run_name
        )

        self.additional_metrics.update(f"{prefix}ausc", ausc)

        # Read ood metrics
        with open(vis_path / "ood_metrics.json", "r") as f:
            ood_metrics = json.load(f)
            self.additional_metrics.update(f"{prefix}auroc", ood_metrics["auroc"])
            self.additional_metrics.update(f"{prefix}auprc", ood_metrics["auprc"])

        # Save metrics
        metrics = self.metrics.get_dict()
        with open(vis_path / "metrics.json", "w") as f:
            json.dump(metrics, f)

        expected_metrics = self.expected_metrics.get_dict()
        with open(vis_path / "expected_metrics.json", "w") as f:
            json.dump(expected_metrics, f)

        additional_metrics = self.additional_metrics.get_dict()
        with open(vis_path / "additional_metrics.json", "w") as f:
            json.dump(additional_metrics, f)

        # Save hparams
        with open(vis_path / "hparams.json", "w") as f:
            json.dump(self.get_hparams(), f)

        # Save additional metrics for tensorboard in log_hyperparams
        add_metrics = [
            f"{prefix}ece",
            f"{prefix}ausc",
            f"{prefix}auroc",
            f"{prefix}auprc",
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
        data_module.prepare_data()
        data_module.setup()

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
            replace_sampler=False if data_module.sampler else True,
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
