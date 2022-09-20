import datetime
import logging
import torch
from matplotlib import pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from src.lightning_modules.BaseLightningModule import BaseLightningModule
from src.baselines.Laplace_online.utils import sample_nn
from src.laplace.hessian.layerwise import ContrastiveHessianCalculator

plt.switch_backend("agg")
logging.getLogger(__name__).setLevel(logging.INFO)


class LaplaceOnlineLightningModule(BaseLightningModule):
    def init(self, model, loss_fn, miner, calculator_cls, optimizer, dataset_size, args):
        super().init(model, loss_fn, miner, optimizer, args)

        self.hessian_memory_factor = args.hessian_memory_factor
        self.data_size = dataset_size
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.hessian = self.data_size**2 * torch.ones_like(
            parameters_to_vector(self.model.linear.parameters())
        ).to(device)
        self.scale = 1
        self.prior_prec = 1

        self.n_train_samples = 1
        self.n_val_samples = 1
        self.n_test_samples = 100

        self.hessian_calculator = calculator_cls(device=self.device, margin=args.margin)
        self.hessian_calculator.init_model(self.model.linear)

    def train_step(self, x, y):

        # pass the data through the deterministic model
        x = self.model.conv(x)

        mu_q = parameters_to_vector(self.model.linear.parameters())

        # get posterior scale
        sigma_q = 1 / (self.hessian * self.scale + self.prior_prec + 1e-8).sqrt()

        samples = sample_nn(mu_q, sigma_q, self.n_train_samples)

        z = []
        hessian = []
        loss_running_sum = 0
        for sample in samples:
            # use sampled samples to compute the loss
            vector_to_parameters(sample, self.model.linear.parameters())

            zs = self.model.linear(x)

            hard_pairs = self.miner(zs, y)
            loss_running_sum = self.loss_fn(zs, y, indices_tuple=hard_pairs)

            # compute hessian
            h_s = self.hessian_calculator.compute_batch_pairs(hard_pairs)

            # scale from batch to dataset size
            scale = self.data_size**2 / (len(hard_pairs[0]) + len(hard_pairs[2])) ** 2
            h_s = torch.clamp(h_s * scale, min=0)

            # append results
            hessian.append(h_s)
            z.append(zs)

        loss = loss_running_sum / self.n_train_samples
        hessian = torch.stack(h_s).mean(0) if len(hessian) > 1 else h_s
        z_mu = torch.stack(z).mean(0) if len(z) > 1 else zs

        # update hessian
        self.hessian = self.hessian_memory_factor * self.hessian + hessian

        # put mean parameter as before
        vector_to_parameters(mu_q, self.model.linear.parameters())

        self.metrics.update("train_loss", loss.item())

        return z_mu, loss

    def forward_samples(self, x, n_samples):

        # pass the data through the deterministic model
        x = self.model.conv(x)

        # get posterior scale
        sigma_q = 1 / (self.hessian * self.scale + self.prior_prec + 1e-8).sqrt()
        mu_q = parameters_to_vector(self.model.linear.parameters())

        samples = sample_nn(mu_q, sigma_q, n_samples)

        z = []
        for sample in samples:
            # use sampled samples to compute the loss
            vector_to_parameters(sample, self.model.linear.parameters())

            zs = self.model.linear(x)
            z.append(zs)

        if len(z) > 1:
            z = torch.stack(z)
            z_mu = z.mean(0)
            z_sigma = z.std(0)
        else:
            z_mu = zs
            z_sigma = torch.zeros_like(z_mu)

        # put mean parameter as before
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return z_mu, z_sigma, z

    def val_step(self, x, y):

        z_mu, z_sigma, z = self.forward_samples(x, self.n_val_samples)

        # evaluate mean metrics
        hard_pairs = self.miner(z_mu, y)
        loss = self.loss_fn(z_mu, y, indices_tuple=hard_pairs)
        self.metrics.update("val_loss", loss.item())

        return z_mu, z_sigma, z

    def test_step(self, x, y):

        z_mu, z_sigma, z = self.forward_samples(x, self.n_val_samples)
        return z_mu, z_sigma, z

    def ood_step(self, x, y):

        z_mu, z_sigma, _ = self.forward_samples(x, self.n_val_samples)
        return z_mu, z_sigma
