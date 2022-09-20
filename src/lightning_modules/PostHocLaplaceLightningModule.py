from torch import optim, nn
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tqdm import tqdm

from src.utils import filter_state_dict
from src.baselines.Laplace_online.utils import sample_nn
from src.baselines.Laplace_posthoc.utils import optimize_prior_precision
from src.lightning_modules.BaseLightningModule import BaseLightningModule
from matplotlib import pyplot as plt
from pathlib import Path

class DummyOptimizer(optim.Optimizer):
    def __init__(self, lr=1e-3):
        defaults = dict(lr=lr)
        params = [{"params": [torch.randn(1, 1)], "lr": lr}]
        super(DummyOptimizer, self).__init__(params, defaults)

    def step(self):
        pass


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class PostHocLaplaceLightningModule(BaseLightningModule):
    """
    Post-hoc Laplace approximation of the posterior distribution.
    """

    def init(self, model, miner, calculator_cls, dataset_size, args):
        super().init(model, DummyLoss(), miner, DummyOptimizer(), args)

        self.data_size = dataset_size

        # Load model backbone
        if args.backbone_path:
            state_dict = torch.load(args.backbone_path)
            model.load_state_dict(
                filter_state_dict(state_dict, remove="module.0.")
            )

        self.hessian_calculator = calculator_cls(device=self.device, margin=args.margin)
        self.hessian_calculator.init_model(self.model.linear)

        self.n_test_samples = args.posterior_samples

    def forward(self, x):

        x = self.model.conv(x)
        x = self.model.linear(x)

        return x

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
            z = torch.stack(z, dim=0)
            z_mu = zs
            z_sigma = torch.zeros_like(z_mu)

        # put mean parameter as before
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return z_mu, z_sigma, z

    def ood_step(self, X, y):
        mean, std, samples = self.forward_samples(X, self.n_test_samples)
        return mean, std, samples

    def test_step(self, X, y):
        mean, std, samples = self.forward_samples(X, self.n_test_samples)
        return mean, std, samples

    def train_start(self):
        self.epoch = 0

    def train(self):
        """Specialized train loop for post hoc"""

        self.train_start()
        print("Starting post hoc optimization")

        if not self.name:
            raise ValueError("Please run .init()")
        
        hessian = torch.zeros_like(
            parameters_to_vector(self.model.linear.parameters()), device=self.device
        )
        self.model.eval()
        with torch.inference_mode():
            for x, y in tqdm(self.train_loader):
                output = self.forward(x)
                hard_pairs = self.miner(output, y)

                # compute hessian
                h_s = self.hessian_calculator.compute_batch_pairs(hard_pairs)

                # scale from batch size to data size
                scale = self.data_size**2 / (2 * len(hard_pairs[0]) + 2 * len(hard_pairs[2]))
                hessian += torch.clamp(h_s * scale, min=0)

        # Scale by number of batches
        hessian /= len(self.train_loader)

        print(
            f"{100 * self.hessian_calculator.zeros / self.hessian_calculator.total_pairs:.2f}% of pairs are zero."
        )
        print(
            f"{100 * self.hessian_calculator.negatives / self.hessian_calculator.total_pairs:.2f}% of pairs are negative."
        )

        self.visualize_hessian(hessian, "hessian_before")
        self.visualize_hessian(hessian + 1, "precision_before")
        self.visualize_hessian(1 / (hessian + 1), "posterior_before")
        
        mu_q = parameters_to_vector(self.model.linear.parameters())

        scale = 1.0
        prior_prec = 1.0
        prior_prec = torch.tensor(prior_prec)
        prior_prec = optimize_prior_precision(mu_q, hessian, prior_prec)
        print("prior precision is ==>> ", prior_prec)
        self.visualize_hessian(hessian + prior_prec, "precision_after")
        self.visualize_hessian( 1 / (hessian + prior_prec), "posterior_after")

        self.hessian = hessian
        self.prior_prec = prior_prec
        self.scale = scale

        print("Finished optimizing post-hoc")


    def visualize_hessian(self, hessian, name):
        
        # Set path
        vis_path = (
            Path(self.args.vis_dir)
            / self.args.dataset
            / self.name
            / f"epoch_{self.epoch + 1}"
        )
        
        vis_path.mkdir(parents=True, exist_ok=True)
        
        vis_path = vis_path / f"{name}.png"
        
        plt.plot(hessian.cpu().numpy())
        plt.yscale("log")
        plt.savefig(vis_path)
        plt.close(); plt.cla(); plt.clf();
        