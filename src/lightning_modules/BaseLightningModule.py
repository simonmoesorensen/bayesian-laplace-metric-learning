import datetime
import json
import logging
import time
from pathlib import Path, PosixPath

import torch
from matplotlib import pyplot as plt
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.evaluate import compute_map_k, compute_recall_k, compute_rank, compute_pidx
from src.visualize import visualize, get_vis_path
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
        self.seed_everything(args.random_seed)

        self.args = args
        self.ks = [1, 5, 10, 20]
        
        # LOGGING
        self.name = args.name
        self.name = self.name + "_linear" if args.linear else self.name + "_conv"
        self.writer = self.setup_logger(self.name)

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

        self.epoch = 0
        
        self.n_train_samples = 1
        self.n_val_samples = 10
        self.n_test_samples = 100

    def setup_logger(self, name):
        subdir = get_time()
        logdir = Path(self.args.log_dir) / self.args.dataset / name / subdir
        writer = SummaryWriter(logdir)
        return writer

    def train_step(self, X, pairs):
        raise NotImplementedError()

    def val_step(self, X, y, n_samples=1):
        raise NotImplementedError()

    def test_step(self, X, y, n_samples=1):
        raise NotImplementedError()

    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def train_start(self):
        pass

    def train_end(self):
        pass

    def val_start(self):
        pass

    def val_end(self):
        pass

    def test_start(self):
        pass

    def test_end(self):
        pass

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

        disp_freq = (
            len(self.train_loader) // self.args.disp_freq
        )  # frequency to display training loss
        self.batch = 0

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
                if ((self.batch + 1) % disp_freq == 0) and self.batch != 0:
                    
                    #TODO: maybe we want also to evaluate on training set?
                    #TODO: we could evaluate just on batch...    
                    
                    self.writer.add_scalar("train/loss", loss.item(), self.batch)
                    self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.batch)

                self.batch += 1

            self.epoch_end()

            # Validate @ frequency
            if (epoch + 1) % self.args.save_freq == 0:
                print("=" * 60, flush=True)
                metrics = self.validate()
                self.save_model()
                print("=" * 60, flush=True)
                
                for k in metrics:
                    self.writer.add_scalar(f"val/{k}", metrics[k], epoch)

        self.train_end()
        print(f"Finished training @ epoch: {self.epoch + 1}")
        return self.model

    def compute_features(self, loader, n_samples=1):
        self.model.eval()

        z_mu = []
        z_sigma = []
        z_samples = []
        labels = []
        images = []
        with torch.no_grad():
            for image, target in tqdm(loader, desc="Computing features"):
                mu, sigma, samples = self.test_step(image, target, n_samples=n_samples)
                z_mu.append(mu.cpu())
                if sigma is not None:
                    z_sigma.append(sigma.cpu())
                if samples is not None:
                    z_samples.append(samples.cpu())
                labels.append(target.cpu())
                images.append(image.cpu())
                
        z_mu = torch.cat(z_mu, dim=0)
        
        if len(z_sigma) > 0:
            z_sigma = torch.cat(z_sigma, dim=0)

        if len(z_samples) > 0:    
            z_samples = torch.cat(z_samples, dim=1).permute(1, 0, 2)
            
        labels = torch.cat(labels, dim=0)
        images = torch.cat(images, dim=0)

        return z_mu, z_sigma, z_samples, labels, images

    def validate(self):
        print(f"Validating @ epoch: {self.epoch + 1}")

        self.val_start()
        self.model.eval()

        z_mu, z_sigma, z_samples, labels, images = self.compute_features(self.val_loader, n_samples=self.n_val_samples)
        ood_z_mu, ood_z_sigma, ood_z_samples, ood_labels, ood_images  = self.compute_features(self.ood_loader, n_samples=self.n_val_samples)
        
        pos_idx = compute_pidx(labels.cpu().numpy())
        rank = compute_rank(z_mu.numpy(), None, samesource=True)
        mapk = [compute_map_k(rank, pos_idx, k) for k in self.ks]
        recallk = compute_recall_k(rank, pos_idx, self.ks)
        
        if self.to_visualize:
            dict_ = {"z_mu": z_mu, 
                     "z_sigma": z_sigma, 
                     "z_samples": z_samples, 
                     "labels": labels, 
                     "images": images}
            dict_ood = {"z_mu": ood_z_mu, 
                        "z_sigma": ood_z_sigma, 
                        "z_samples": ood_z_samples, 
                        "labels": ood_labels, 
                        "images": ood_images}
            dict_other = {}
            if hasattr(self, "hessian") :
                dict_other["hessian"] = self.hessian
            dict_log = {"path": self.args.vis_dir,
                        "dataset": self.args.dataset,
                        "name": self.name,
                        "epoch": self.epoch +1}
            
            metrics = visualize(
                dict_, 
                dict_ood, 
                dict_other,
                dict_log, 
                prefix="val",
            )
            
        for i, k in enumerate(self.ks):
            metrics[f"map{k}"] = mapk[i]
            metrics[f"recall{k}"] = recallk[i]
            
        # Save metrics
        vis_path = get_vis_path(dict_log)
        with open(vis_path / "val_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        # Save hparams
        with open(vis_path / "hparams.json", "w") as f:
            json.dump(self.get_hparams(), f)

        self.val_end()
        
        return metrics

    def test(self):
        print(f"Testing @ epoch: {self.epoch + 1}")
        self.model.eval()

        self.test_start()
        
        z_mu, z_sigma, z_samples, labels, images = self.compute_features(self.test_loader, n_samples=self.n_test_samples)
        ood_z_mu, ood_z_sigma, ood_z_samples, ood_labels, ood_images  = self.compute_features(self.ood_loader, n_samples=self.n_test_samples)
                
        pos_idx = compute_pidx(labels.cpu().numpy())
        rank = compute_rank(z_mu.numpy(), None, samesource=True)
        mapk = [compute_map_k(rank, pos_idx, k) for k in self.ks]
        recallk = compute_recall_k(rank, pos_idx, self.ks)
        
        #TODO: implemented Sørens metrics based on samples
        #TODO: implemnented distribution on distances
            
        self.test_end()
        
        dict_ = {"z_mu": z_mu, 
                    "z_sigma": z_sigma, 
                    "z_samples": z_samples, 
                    "labels": labels, 
                    "images": images}
        dict_ood = {"z_mu": ood_z_mu, 
                    "z_sigma": ood_z_sigma, 
                    "z_samples": ood_z_samples, 
                    "labels": ood_labels, 
                    "images": ood_images}
        dict_other = {}
        if hasattr(self, "hessian") :
            dict_other["hessian"] = self.hessian
        dict_log = {"path": self.args.vis_dir,
                    "dataset": self.args.dataset,
                    "name": self.name,
                    "epoch": self.epoch +1}
        
        metrics = visualize(
            dict_, 
            dict_ood, 
            dict_other,
            dict_log, 
            prefix="test",
        )
        
        for i, k in enumerate(self.ks):
            metrics[f"map{k}"] = mapk[i]
            metrics[f"recall{k}"] = recallk[i]
            
        # Save metrics
        vis_path = get_vis_path(dict_log)
        with open(vis_path / "test_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        # Save hparams
        with open(vis_path / "hparams.json", "w") as f:
            json.dump(self.get_hparams(), f)


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

        hparams = self.get_hparams()
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={},
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
            / self.name
            / name
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model @ {str(path)}")
        self.save(content=self.model.module.state_dict(), filepath=str(path))
        
        if hasattr(self, "hessian"):
            torch.save(self.hessian, path.parent / "hessian.pth")
