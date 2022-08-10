import datetime
import json
import logging
from pathlib import Path, PosixPath
import time

import torch
from matplotlib import pyplot as plt
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.visualize import visualize_all
from src.metrics.MetricMeter import MetricMeter, AverageMeter

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)
torch.manual_seed(1234)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class BaseLightningModule(LightningLite, MetricMeter):
    """
    Base Lightning Module for Probabilistic Metric Learning
    """

    def init(self, model, loss_fn, miner, optimizer, args):
        print("Overall Configurations:")
        print("=" * 60)
        for k in args.__dict__:
            print(" '{}' : '{}' ".format(k, str(args.__dict__[k])))

        # Learning rate scheduler options
        base_lr = args.lr
        max_lr = min(args.lr * 1e3, 0.1)
        # Cycle every 5% of total epochs, results in base_lr around 60% of total epochs
        # See https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling?scriptVersionId=38549725&cellId=17
        step_size_up = max(1, args.num_epoch // 20)

        print(" 'base_lr' : '{}' ".format(base_lr))
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
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode="triangular2",
            cycle_momentum=False,
        )

        # Metric calculation
        self.metric_calc = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1"),
            k="max_bin_count",
            device=self.device,
        )

        # Meters
        self.metrics = MetricMeter(
            meters={
                "train_accuracy": AverageMeter(),
                "train_map_r": AverageMeter(),
                "val_accuracy": AverageMeter(),
                "val_map_r": AverageMeter(),
                "test_accuracy": AverageMeter(),
                "test_map_r": AverageMeter(),
                "train_loss": AverageMeter(),
                "val_loss": AverageMeter(),
            },
            batch_size=self.batch_size,
        )

    def setup_logger(self, name):
        subdir = get_time()
        logdir = Path(self.args.log_dir) / name / subdir
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

    def train_step(self, X, y):
        raise NotImplementedError()

    def val_step(self, X, y):
        raise NotImplementedError()

    def test_step(self, X, y):
        raise NotImplementedError()

    def ood_step(self, X, y):
        raise NotImplementedError()

    def epoch_start(self):
        self.metrics.reset(["train_loss", "train_accuracy", "train_map_r"])

    def epoch_end(self):
        self.log(["train_loss", "train_accuracy", "train_map_r"])

    def train_start(self):
        pass

    def train_end(self):
        pass

    def val_start(self):
        self.metrics.reset(["val_loss", "val_accuracy", "val_map_r"])

    def val_end(self):
        self.log(["val_loss", "val_accuracy", "val_map_r"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Validation MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("val_loss"),
                acc=self.metrics.get("val_accuracy"),
                map_r=self.metrics.get("val_map_r"),
            ),
            flush=True,
        )

    def test_start(self):
        self.metrics.reset(["test_accuracy", "test_map_r"])

    def test_end(self):
        self.log(["test_accuracy", "test_map_r"])

        # display training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Test Accuracy {acc.val:.4f} ({acc.avg:.4f})\t"
            "Test MAP@r {map_r.val:.4f} ({map_r.avg:.4f}))".format(
                time.asctime(time.localtime(time.time())),
                acc=self.metrics.get("test_accuracy"),
                map_r=self.metrics.get("test_map_r"),
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
            "Training MAP@r {map_r.val:.4f} ({map_r.avg:.4f})\t"
            "Lr {lr:.4f}".format(
                epoch + 1,
                self.args.num_epoch,
                batch + 1,
                len(self.train_loader) * self.args.num_epoch,
                time.asctime(time.localtime(time.time())),
                loss=self.metrics.get("train_loss"),
                acc=self.metrics.get("train_accuracy"),
                map_r=self.metrics.get("train_map_r"),
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

    def update_accuracy(self, z, y, step="train"):
        if step not in ["train", "val", "test"]:
            raise ValueError("step must be one of ['train', 'val', 'test']")

        # Metrics
        with torch.no_grad():
            metrics = self.metric_calc.get_accuracy(
                query=z,
                reference=z,
                query_labels=y,
                reference_labels=y,
                embeddings_come_from_same_source=True,
            )

            self.metrics.update(f"{step}_accuracy", metrics["precision_at_1"])
            self.metrics.update(f"{step}_map_r", metrics["mean_average_precision_at_r"])

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

            for image, target in tqdm(self.train_loader, desc="Training"):
                self.optimizer.zero_grad()

                out, loss = self.train_step(image, target)

                self.backward(loss)

                self.optimizer_step()

                # display and log metrics every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    self.update_accuracy(out, target, "train")
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

        id_sigma = []
        id_mu = []
        id_images = []
        with torch.no_grad():
            for image, target in tqdm(self.val_loader, desc="Validating"):
                mu, sigma, out = self.val_step(image, target)
                id_sigma.append(sigma)
                id_mu.append(mu)
                id_images.append(image)

                self.update_accuracy(out, target, "val")

        self.val_end()

        if self.to_visualize:
            self.visualize(id_mu, id_sigma, id_images, prefix="val_")

        self.model.train()

    def test(self):
        print(f"Testing @ epoch: {self.epoch + 1}")
        self.model.eval()

        self.test_start()

        id_sigma = []
        id_mu = []
        id_images = []
        with torch.no_grad():
            for image, target in tqdm(self.test_loader, desc="Testing"):
                mu, sigma, out = self.test_step(image, target)
                id_sigma.append(sigma)
                id_mu.append(mu)
                id_images.append(image)

                self.update_accuracy(out, target, "test")

        self.test_end()

        if self.to_visualize:
            self.visualize(id_mu, id_sigma, id_images, prefix="test_")

    def visualize(self, id_mu, id_sigma, id_images, prefix):
        print("=" * 60, flush=True)
        print("Visualizing...")

        # Set path
        vis_path = Path(self.args.vis_dir) / self.name / f"epoch_{self.epoch + 1}"
        vis_path.mkdir(parents=True, exist_ok=True)

        ood_sigma = []
        ood_mu = []
        ood_images = []
        for img, y in tqdm(self.ood_loader, desc="OOD"):
            mu_dul, std_dul = self.ood_step(img, y)
            ood_sigma.append(std_dul)
            ood_mu.append(mu_dul)
            ood_images.append(img)

        visualize_all(
            id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
        )

        # Save metrics
        with open(vis_path / "metrics.json", "w") as f:
            json.dump(self.metrics.get_dict(), f)

    def forward(self, x):
        return self.model(x)

    def log_hyperparams(self):
        print("Logging hyperparameters")

        hparams = vars(self.args)
        hparams["name"] = self.name
        hparams["epoch"] = self.epoch
        hparams["miner"] = self.miner.__class__.__name__
        hparams["model"] = self.model.module.module.__class__.__name__
        hparams["optimizer"] = self.optimizer.__class__.__name__
        hparams["loss_fn"] = self.loss_fn.__class__.__name__

        for key, val in hparams.items():
            if isinstance(val, PosixPath):
                hparams[key] = str(val)
            elif isinstance(val, list):
                hparams[key] = str(val)

        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={
                "train_accuracy": self.metrics.get("train_accuracy").avg,
                "train_map_r": self.metrics.get("train_map_r").avg,
                "val_accuracy": self.metrics.get("val_accuracy").avg,
                "val_map_r": self.metrics.get("val_map_r").avg,
                "test_accuracy": self.metrics.get("test_accuracy").avg,
                "test_map_r": self.metrics.get("test_map_r").avg,
            },
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

        path = Path(self.args.model_save_folder) / self.args.name / name

        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model @ {str(path)}")
        self.save(content=self.model.module.module.state_dict(), filepath=str(path))
