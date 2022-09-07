from src.lightning_modules.BaseLightningModule import BaseLightningModule
from torch import optim, nn
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector

from tqdm import tqdm

from src.utils import filter_state_dict


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

    def init(self, model, miner, calculator_cls, inference_model, args):
        super().init(model, DummyLoss(), miner, DummyOptimizer(), args)

        # Load model backbone
        if args.backbone_path:
            state_dict = torch.load(args.backbone_path)
            model.backbone.load_state_dict(
                filter_state_dict(state_dict, remove="module.0.")
            )

        self.calculator = calculator_cls(device=self.device, margin=args.margin)

        if inference_model is None:
            self.inference_model = model
        else:
            self.inference_model = inference_model

        self.calculator.init_model(self.inference_model)

        self.n_samples = args.posterior_samples

        self.model.module.module.inference_model = self.inference_model

    def forward(self, x, use_samples=True):
        return self.model(x, use_samples=use_samples)

    def ood_step(self, X, y):
        mean, std, samples = self.forward(X, use_samples=True)
        return mean, std, samples

    def test_start(self):
        super().test_start()
        self.generate_nn_samples()

    def generate_nn_samples(self):
        # Use the same samples for testing
        self.model.module.module.generate_nn_samples(
            self.mu_q, self.sigma_q, self.n_samples
        )

    def test_step(self, X, y):
        mean, std, samples = self.forward(X, use_samples=True)

        return mean, std, samples

    def train_start(self):
        self.epoch = 0

    def train(self):
        """Specialized train loop for post hoc"""
        self.model.train()
        self.train_start()

        print("Starting post hoc optimization")

        if not self.name:
            raise ValueError("Please run .init()")

        h = 0
        dataset_size = len(self.train_loader.dataset)
        print(f"Dataset size: {dataset_size}")
        with torch.no_grad():
            for x, y in tqdm(self.train_loader):
                output = self.forward(x, use_samples=False)
                hard_pairs = self.miner(output, y)

                # Total number of possible pairs / number of pairs in our batch
                scaler = dataset_size**2 / x.shape[0] ** 2
                hessian = self.calculator.compute_batch_pairs(hard_pairs)
                h += hessian * scaler

        if (h < 0).sum():
            print("Found negative values in Hessian.")

        # Scale by number of batches
        h /= len(self.train_loader)

        h = torch.maximum(h, torch.tensor(0))

        print(
            f"{100 * self.calculator.zeros / self.calculator.total_pairs:.2f}% of pairs are zero."
        )
        print(
            f"{100 * self.calculator.negatives / self.calculator.total_pairs:.2f}% of pairs are negative."
        )

        map_solution = parameters_to_vector(self.inference_model.parameters())

        scale = 1.0
        prior_prec = 1.0
        prior_prec = self.optimize_prior_precision(
            map_solution, h, torch.tensor(prior_prec)
        )
        posterior_precision = h * scale + prior_prec
        posterior_scale = 1.0 / (posterior_precision.sqrt() + 1e-6)

        self.mu_q = map_solution
        self.sigma_q = posterior_scale
        self.h = h

        print("Finished optimizing post-hoc")

    def scatter(self, mu_q, prior_precision_diag):
        return (mu_q * prior_precision_diag) @ mu_q

    def log_det_ratio(self, hessian, prior_prec):
        posterior_precision = hessian + prior_prec
        log_det_prior_precision = len(hessian) * prior_prec.log()
        log_det_posterior_precision = posterior_precision.log().sum()
        return log_det_posterior_precision - log_det_prior_precision

    def log_marginal_likelihood(self, mu_q, hessian, prior_prec):
        # we ignore neg log likelihood as it is constant wrt prior_prec
        neg_log_marglik = -0.5 * (
            self.log_det_ratio(hessian, prior_prec) + self.scatter(mu_q, prior_prec)
        )
        return neg_log_marglik

    def optimize_prior_precision(self, mu_q, hessian, prior_prec, n_steps=100):

        log_prior_prec = prior_prec.log()
        log_prior_prec.requires_grad = True
        optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
        for _ in range(n_steps):
            optimizer.zero_grad()
            prior_prec = log_prior_prec.exp()
            neg_log_marglik = -self.log_marginal_likelihood(mu_q, hessian, prior_prec)
            neg_log_marglik.backward()
            optimizer.step()

        prior_prec = log_prior_prec.detach().exp()

        return prior_prec
