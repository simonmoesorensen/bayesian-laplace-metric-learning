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

    def loss_step(self, mu, std, y, step, n_samples=1):
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
        cov = torch.diag_embed(std.square())
        pdist = tdist.MultivariateNormal(mu, cov)

        # Compare to unit gaussian r(z) ~ N(0, I)
        zdist = tdist.MultivariateNormal(
            torch.zeros_like(mu),
            torch.diag_embed(torch.ones_like(std)),
        )

        # Monte Carlo K times sampling, reparameterization trick in order
        # to do backprop
        # [K_samples, batch_size, embedding_space]
        samples = pdist.rsample([n_samples])

        # Repeat interleave tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[1,2],[3,4],[3,4],[4,5],[4,5]]
        k1_samples = samples.repeat_interleave(n_samples, dim=0)

        # Repeat tensor so something like [[1,2],[3,4],[4,5]] becomes
        # [[1,2],[3,4],[4,5],[1,2],[3,4],[4,5]]
        k2_samples = samples.repeat(n_samples, 1, 1)

        # Convert 3D to 2D, we use self.K**3 because we have K_samples, repeated K times
        # Concatenates all K samples into one large
        # [batch_size * K, embedding_space] tensor
        k1_samples = k1_samples.view(
            n_samples**2 * mu.shape[0], self.args.embedding_size
        )
        k2_samples = k2_samples.view(
            n_samples**2 * mu.shape[0], self.args.embedding_size
        )

        # Repeat target to match the shape of the samples
        # [batch_size * K, embedding_space]
        y_k1 = y.repeat_interleave(n_samples**2, dim=0)
        y_k2 = y.repeat(n_samples**2, 1).view(-1)

        # See hib-pair-indicies.ipynb for more info
        # Scale the indices to match the shape of the samples
        def scale_indices(x):
            step = self.to_device(
                torch.arange(0, n_samples**2).repeat_interleave(x.shape[0]) * mu.shape[0]
            )

            return x.repeat(n_samples**2) + step

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

        return samples, loss

    def train_step(self, X, y):
        # Pass images through the model
        mu, std = self.forward(X)

        samples, loss = self.loss_step(mu, std, y, step="train", n_samples=self.n_train_samples)

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
