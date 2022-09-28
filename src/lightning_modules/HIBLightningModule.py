import datetime
import logging
import time
from pathlib import Path

import torch
import torch.distributions as tdist
from matplotlib import pyplot as plt
from src.lightning_modules.BaseLightningModule import BaseLightningModule
from tqdm import tqdm

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")


class HIBLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, optimizer, args):
        super().init(model, loss_fn, miner, optimizer, args)

        loss_path = None

        if args.model_path and not args.loss_path:
            loss_path = args.model_path.replace("Model", "Loss", 1)

        elif args.model_path and args.loss_path:
            loss_path = args.loss_path

        elif not args.model_path and args.loss_path:
            raise Exception(
                "You can not specify a loss path without a model path!"
                " Use --model_path to specify the model path."
            )

        if loss_path is not None:
            state_dict = self.load(loss_path)

            new_state_dict = {}
            for key in state_dict:
                if key.startswith("module."):
                    new_state_dict[key[7:]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]

            loss_fn.load_state_dict(new_state_dict)

        # REQUIRED FOR SOFT CONTRASTIVE LOSS
        self.loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.001)

        # Move loss fn parameters to GPU
        self.loss_fn.cast_params(self.device)

    def optimizer_step(self):
        self.optimizer.step()
        self.loss_optimizer.step()

        self.loss_fn.apply(self.loss_fn.weight_clipper)

    def loss_step(self, mu, std, pairs, step, n_samples=1):
        
        ap, pos, an, neg = pairs
        
        loss_pos = 0
        for _ in range(n_samples):
            z_ap = mu[ap] + std[ap] * torch.randn_like(std[ap])
            for _ in range(n_samples):
                z_pos = mu[pos] + std[pos] * torch.randn_like(std[pos])
                pair_dist = torch.norm(z_ap - z_pos, dim=1)
                loss_pos += torch.sigmoid(-self.model.A * pair_dist + self.model.B).mean()
        
        loss_neg = 0
        for _ in range(n_samples):
            z_an = mu[an] + std[an] * torch.randn_like(std[an])
            for _ in range(n_samples):
                z_neg = mu[neg] + std[neg] * torch.randn_like(std[neg])
                pair_dist = torch.norm(z_an - z_neg, dim=1)
                loss_neg += torch.sigmoid(-self.model.A * pair_dist + self.model.B).mean()

        loss_kl = (- 0.5 * torch.sum(1 + torch.log(std**2) - mu.square() - std**2)).mean()

        loss = loss_pos + loss_neg + self.args.kl_scale * loss_kl
        
        samples = mu + std * torch.randn_like(std)
        
        return samples, loss

    def train_step(self, X, pairs, class_labels):
        # Pass images through the model
        mu, std = self.forward(X)

        samples, loss = self.loss_step(mu, std, pairs, step="train", n_samples=self.n_train_samples)

        return samples[0], loss

    def val_step(self, X, y, n_samples=1):
        mu, std = self.forward(X)

        samples, _ = self.loss_step(mu, std, y, step="val", n_samples=n_samples)

        return mu, std, samples[0]

    def test_step(self, X, y, n_samples=1):
        mu, std = self.forward(X)

        # Reparameterization trick
        cov = torch.diag_embed(std ** 2)
        pdist = tdist.MultivariateNormal(mu, cov)
        samples = pdist.rsample([n_samples])

        return mu, std, samples


    def save_model(self, prefix=None):
        current_time = get_time()

        model_name = "Model_Epoch_{}_Time_{}_checkpoint.pth".format(
            self.epoch + 1, current_time
        )

        if prefix is not None:
            model_name = prefix + "_" + model_name

        path = Path(self.args.model_save_folder) / self.args.dataset / self.args.name
        model_path = path / model_name

        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model @ {str(path)}")
        self.save(
            content=self.model.module.module.state_dict(), filepath=str(model_path)
        )

        loss_name = "Loss_Epoch_{}_Time_{}_checkpoint.pth".format(
            self.epoch + 1, current_time
        )

        if prefix is not None:
            loss_name = prefix + "_" + loss_name

        loss_path = path / loss_name

        print(f"Saving loss @ {str(path)}")
        torch.save(self.loss_fn.state_dict(), loss_path)

    def load(self, filepath):
        return self._strategy.load_checkpoint(filepath)
