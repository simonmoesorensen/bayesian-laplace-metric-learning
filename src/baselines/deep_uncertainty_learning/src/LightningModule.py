import datetime
import logging
from pathlib import Path
import time

import torch
from matplotlib import pyplot as plt
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, l2_norm, test_model

plt.switch_backend("agg")
logging.getLogger().setLevel(logging.INFO)
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
        self.miner = miner

        # Lite setup
        self.model, self.optimizer = self.setup(model, optimizer)

    def setup_logger(self, name):
        subdir = get_time()
        logdir = f"{self.dul_args.log_dir}/{name}/{subdir}"
        writer = SummaryWriter(logdir)
        return writer

    # noinspection PyMethodOverriding
    def run(self):

        logging.info(f"Training")
        self.model.train()

        if not self.name:
            raise ValueError("Please run .init()")

        losses = AverageMeter()
        losses_KL = AverageMeter()
        DISP_FREQ = len(self.train_loader) // 20  # frequency to display training loss
        batch = 0

        for epoch in range(self.dul_args.num_epoch):
            if epoch < self.dul_args.resume_epoch:
                continue

            self.epoch = epoch

            for i, (image, target) in enumerate(tqdm(self.train_loader)):
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

                losses_KL.update(loss_kl.item(), image.size(0))
                losses.update(loss_backbone.data.item(), image.size(0))

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60, flush=True)
                    print(
                        "Epoch {}/{} Batch (Step) {}/{}\t"
                        "Time {}\t"
                        "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t)".format(
                            epoch + 1,
                            self.dul_args.num_epoch,
                            batch + 1,
                            len(self.train_loader) * self.dul_args.num_epoch,
                            time.asctime(time.localtime(time.time())),
                            loss=losses,
                            loss_KL=losses_KL,
                        ),
                        flush=True,
                    )

                batch += 1

            self.writer.add_scalar(
                "train_loss", losses.avg, global_step=epoch, new_style=True
            )
            self.writer.add_scalar(
                "train_loss_KL", losses_KL.avg, global_step=epoch, new_style=True
            )

            # Validate @ frequency
            if (epoch + 1) % self.dul_args.save_freq == 0:
                print("=" * 60, flush=True)
                self.validate()

                backbone_path = Path(
                    self.dul_args.model_save_folder
                ) / self.dul_args.name / "Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    epoch + 1, batch, get_time()
                )

                backbone_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Saving model @ {str(backbone_path)}")
                self.save(
                    content=self.model.module.state_dict(), filepath=str(backbone_path)
                )
                print("=" * 60, flush=True)


        logging.info(f"Finished training @ epoch: {self.epoch + 1}")
        return self.model

    def validate(self):
        logging.info(f"Validating @ epoch: {self.epoch + 1}")

        self.model.eval()

        val_loss = AverageMeter()
        val_loss_KL = AverageMeter()

        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                mu_dul, std_dul = self.model(image)

                epsilon = torch.randn_like(std_dul)
                samples = mu_dul + epsilon * std_dul
                variance_dul = std_dul**2

                hard_pairs = self.miner(samples, target)
                loss = self.loss_fn(samples, target, hard_pairs)

                loss_kl = (
                    ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
                    .sum(dim=-1)
                    .mean()
                )

                val_loss.update(loss.item(), image.size(0))
                val_loss_KL.update(loss_kl.data.item(), image.size(0))

        self.writer.add_scalar(
            "val_loss", val_loss.avg, global_step=self.epoch, new_style=True
        )
        self.writer.add_scalar(
            "val_loss_KL", val_loss_KL.avg, global_step=self.epoch, new_style=True
        )

        accuracy = test_model(
            self.train_loader.dataset, self.val_loader.dataset, self.model, self.device,
            self.dul_args.batch_size, self.dul_args.num_workers
        )

        self.writer.add_scalar("val_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar(
            "val_map", accuracy["mean_average_precision"], self.epoch
        )

        # dispaly training loss & acc every DISP_FREQ
        print(
            "Time {}\t"
            "Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Validation Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
            "Validation Prec@1 {top1:.3f}\t"
            "Validation MAP {MAP:.3f}".format(
                time.asctime(time.localtime(time.time())),
                loss=val_loss,
                loss_KL=val_loss_KL,
                top1=accuracy["precision_at_1"],
                MAP=accuracy["mean_average_precision"],
            ),
            flush=True,
        )

        if self.to_visualize:
            self.visualize(
                self.val_loader, self.val_loader.dataset.dataset.class_to_idx
            )

        self.model.train()

    def test(self):
        logging.info(f"Testing @ epoch: {self.epoch}")
        accuracy = test_model(
            self.train_loader.dataset, self.test_loader.dataset, self.model, self.device,
            self.dul_args.batch_size, self.dul_args.num_workers
        )

        self.writer.add_scalar("test_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar(
            "test_map", accuracy["mean_average_precision"], self.epoch
        )

        if self.to_visualize:
            self.visualize(self.test_loader, self.test_loader.dataset.class_to_idx)

    def visualize(self, dataloader, class_to_idx):
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def log_hyperparams(self):
        logging.info("Logging hyperparameters")

        train_accuracy = test_model(
            self.train_loader.dataset,
            self.train_loader.dataset,
            self.model,
            self.device,
            self.dul_args.batch_size,
            self.dul_args.num_workers,
        )
        logging.info(f"{train_accuracy=}")

        val_accuracy = test_model(
            self.train_loader.dataset, self.val_loader.dataset, self.model, self.device,
            self.dul_args.batch_size, self.dul_args.num_workers
        )
        logging.info(f"{val_accuracy=}")

        logging.info("Calculating test accuracy")
        test_accuracy = test_model(
            self.train_loader.dataset, self.test_loader.dataset, self.model, self.device,
            self.dul_args.batch_size, self.dul_args.num_workers
        )
        logging.info(f"{test_accuracy=}")

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

    def add_data(self, data_module):
        data = data_module(
            self.dul_args.data_dir, self.dul_args.batch_size, self.dul_args.num_workers
        )

        data.prepare_data()
        data.setup()
        
        self.train_loader, self.val_loader, self.test_loader = self.setup_dataloaders(
            data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
        )
