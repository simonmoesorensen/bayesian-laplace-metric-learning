import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.lightning_modules.BaseLightningModule import BaseLightningModule


class OnlineLaplaceLightningModule(BaseLightningModule):
    """
    Post-hoc Laplace approximation of the posterior distribution.
    """

    def init(self, model, loss_fn, miner, optimizer, args) -> None:
        """
        Initialize the module.

        :param model: nn.Sequential, the model to train.
        :param loss_fn: nn.Module, the loss function.
        :param miner: Miner, the miner to use.
        :param optimizer: optim.Optimizer, the optimizer to use.
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
        :return:
        """
        super().init(model, loss_fn, miner, optimizer, args)
        self.inference_model = args.inference_model
        self.hessian_calculator = args.hessian_calculator
        self.hessian_calculator.init_model(self.model)
        self.n_posterior_samples: int = args.posterior_samples
        self.compute_hessian_every: int = args.compute_hessian_every
        self.kl_weight: float = args.kl_weight

    def train_start(self):
        self.hessian = torch.tensor(0.0, device=self.device)
        self.scaler: float = len(self.train_loader.dataset) ** 2 / self.batch_size**2

    def on_train_epoch_start(self):
        self.compute_hessian: bool = self.current_epoch % self.compute_hessian_every == 0

    def train_step(self, X, y):
        X, y = X.to(self.device), y.to(self.device)

        self.mu_q = parameters_to_vector(self.inference_model.parameters())
        self.sigma_q = 1 / (self.hessian + 1e-6)

        kl_term = self.compute_kl_term()

        sampled_nn = self._sample_posterior()

        con_loss = 0
        if self.compute_hessian:
            self.hessian = 0

        for nn_i in sampled_nn:
            vector_to_parameters(nn_i, self.inference_model.parameters())
            output = self.model(X)
            hard_pairs = self.miner(output, y)

            if self.compute_hessian:
                # Adjust hessian to the batch size
                hessian_batch = self.hessian_calculator.compute_batch_pairs(self.inference_model, hard_pairs)
                self.hessian += hessian_batch * self.scaler

            con_loss += self.loss_fn(output, y, hard_pairs)

        if self.compute_hessian:
            self.hessian /= self.n_posterior_samples

        con_loss /= self.n_posterior_samples
        loss = con_loss + kl_term.mean() * self.kl_weight
        vector_to_parameters(self.mu_q, self.inference_model.parameters())

        return loss

    def _sample_posterior(self):
        n_params = len(self.mu_q)
        samples = torch.randn(self.n_posterior_samples, n_params, device=self.device)
        samples = samples * self.sigma_q.reshape(1, n_params)
        return self.mu_q.reshape(1, n_params) + samples

    def compute_kl_term(self):
        """
        https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        """
        k = len(self.mu_q)
        return 0.5 * (-torch.log(self.sigma_q) - k + torch.dot(self.mu_q, self.mu_q) + torch.sum(self.sigma_q))
