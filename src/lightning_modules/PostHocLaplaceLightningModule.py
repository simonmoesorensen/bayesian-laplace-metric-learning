import logging
from typing import Tuple
import torch
from torch import Tensor
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.lightning_modules.BaseLightningModule import BaseLightningModule
from src.hessian.layerwise import (
    ContrastiveHessianCalculator,
    FixedContrastiveHessianCalculator,
)


class PostHocLaplaceLightningModule(BaseLightningModule):
    """
    Post-hoc Laplace approximation of the posterior distribution.
    """

    def init(self, model, loss_fn, miner, optimizer, args) -> None:
        """
        Initialize the module.

        :param model: nn.Sequential, the model to train.
        :param miner: Miner, the miner to use.
        :param args: Requires the following arguments:
            - name: str, name of the model
            - dataset: str, dataset name
            - batch_size: int, batch size
            - num_workers: int, number of workers
            - shuffle: bool, whether to shuffle the data
            - pin_memory: bool, whether to use pinned memory
            - neg_margin: float, margin for contrastive loss
            - embedding_size: int, embedding size
            - lr: float, learning rate
            - n_posterior_samples: int, number of posterior samples
            - num_epoch: int, number of epochs
            - resume_epoch: int, resume training from this epoch
            - save_freq: int, save model every save_freq epochs
            - disp_freq: int, display progress every disp_freq batches
            - to_visualize: bool, whether to visualize the model
            - vis_dir: str, path to the visualization directory
            - data_dir: str, path to the data directory
            - model_save_folder: str, path to the model save folder
            - model_path: str, path to the model to load
            - log_dir: str, path to the log directory
        :return: None
        """
        super().init(model, loss_fn, miner, optimizer, args)

        self.n_posterior_samples = args.posterior_samples
        self.scale = 1.0
        self.prior_prec = 1.0
        self.inference_model = getattr(self.model.module, args.inference_model)

        if args.hessian_calculator == "fixed":
            self.hessian_calculator = FixedContrastiveHessianCalculator(
                args.neg_margin, device=self.device
            )
        else:
            self.hessian_calculator = ContrastiveHessianCalculator(
                args.neg_margin, device=self.device
            )
        self.hessian_calculator.init_model(self.inference_model)

    def train_start(self) -> None:
        n_params = sum(p.numel() for p in self.inference_model.parameters())
        self.hessian = torch.zeros((n_params,), device=self.device)
        self.scaler: float = (len(self.train_loader.dataset)**2) / (self.batch_size**2)
        print(self.scaler)

    def train(self) -> None:
        print(f"Training")
        self.train_start()
        self.model.train()

        if not self.name:
            raise ValueError("Please run .init()")

        self.epoch_start()

        for image, target in tqdm(self.train_loader, desc="Training"):
            self.train_step(image, target)

        self.epoch_end()

        # print("=" * 60, flush=True)
        # self.save_model()
        # print("=" * 60, flush=True)

        self.train_end()
        print(f"Finished training")

    def train_step(self, X, y) -> Tensor:
        X = X.to(self.device)
        y = y.to(self.device)

        z = self.model(X)
        hard_pairs = self.miner(z, y)

        hessian = self.hessian_calculator.compute_batch_pairs(hard_pairs)
        self.hessian += self.scaler * hessian
        return hessian

    def train_end(self) -> None:
        if (self.hessian < 0).sum():
            logging.warning("Found negative values in Hessian.")
            self.hessian = self.hessian.clamp(min=0.0)
        
        plt.plot(self.hessian.cpu().numpy())
        plt.savefig("hessian.png")

        self.mu_q: Tensor = parameters_to_vector(self.inference_model.parameters())
        self.sigma_q: Tensor = self.posterior_scale()

    def val_step(self, X, y):
        mean, var = self.predict(X)
        return mean, var, mean

    def test_step(self, X, y):
        mean, var = self.predict(X)
        return mean, var, mean

    def ood_step(self, X, y):
        mean, var = self.predict(X)
        return mean, var

    def predict(self, x: Tensor):
        x = x.to(self.device)
        posterior_samples = self._sample_posterior()

        mean, var = self.generate_predictions_from_samples_rolling(x, posterior_samples)
        return mean, var

    # def predict(self, x: Tensor, agg=torch.mean) -> Tensor:
    #     x = x.to(self.device)
    #     posterior_samples = self._sample_posterior()
    #     preds = []
    #     for sample in posterior_samples:
    #         vector_to_parameters(sample, self.inference_model.parameters())
    #         with torch.inference_mode():
    #             preds.append(self.model(x))
    #     preds = torch.stack(preds)
    #     return agg(preds, dim=0) if agg is not None else preds

    def _sample_posterior(self) -> Tensor:
        n_params = len(self.mu_q)
        samples = torch.randn(self.n_posterior_samples, n_params, device=self.device)
        samples = samples * self.sigma_q.reshape(1, n_params)
        return self.mu_q.reshape(1, n_params) + samples

    def optimize_prior_precision(self, n_steps=100) -> Tensor:
        """
        Optimize the prior precision.

        :param n_steps: int, number of iterations to optimize the prior precision.
        :return: Tensor, the optimized prior precision.
        """
        prior_prec = torch.tensor(1)
        log_prior_prec = prior_prec.log()
        log_prior_prec.requires_grad = True
        optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
        for _ in range(n_steps):
            optimizer.zero_grad()
            prior_prec = log_prior_prec.exp()
            neg_log_marglik = -self.log_marginal_likelihood(prior_prec)
            neg_log_marglik.backward()
            optimizer.step()

        prior_prec = log_prior_prec.detach().exp()
        self.prior_prec = prior_prec

        return prior_prec

    def log_marginal_likelihood(self, prior_prec) -> Tensor:
        # we ignore neg log likelihood as it is constant wrt prior_prec
        neg_log_marglik = -0.5 * (
            _log_det_ratio(self.hessian, prior_prec) + _scatter(self.mu_q, prior_prec)
        )
        return neg_log_marglik

    def posterior_scale(self):
        posterior_precision = self.hessian * self.scale + self.prior_prec
        return 1.0 / (posterior_precision.sqrt() + 1e-6)

    def generate_predictions_from_samples_rolling(
        self, x, weight_samples
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Welford's online algorithm for calculating mean and variance.
        """

        N = len(weight_samples)

        vector_to_parameters(weight_samples[0, :], self.inference_model.parameters())
        with torch.inference_mode():
            mean = self.model(x)
            msq = 0.0
            delta = 0.0

            for i, net_sample in enumerate(weight_samples[1:, :]):
                vector_to_parameters(net_sample, self.inference_model.parameters())
                sample_preds = self.model(x)
                delta = sample_preds - mean
                mean += delta / (i + 1)
                msq += delta * delta

            variance = msq / (N - 1)
        return mean, variance


def _log_det_ratio(hessian, prior_prec) -> Tensor:
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def _scatter(mu_q, prior_precision_diag) -> Tensor:
    return (mu_q * prior_precision_diag) @ mu_q
