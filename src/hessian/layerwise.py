import gc
from abc import abstractmethod
from typing import List

import torch
from torch import Tensor, nn
from torch.nn.utils.convert_parameters import parameters_to_vector

from src.layers import L2Normalize


class HessianCalculator:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, device) -> None:
        self.device = device

    @abstractmethod
    def compute_batch(self, *args, **kwargs):
        pass

    def init_model(self, model):
        self.model = model

        self.feature_maps = []
        self.handles = []

        def fw_hook_get_latent(module, input, output):
            self.feature_maps.append(output.detach())

        def fw_hook_get_input(module, input, output):
            self.feature_maps = [input[0].detach()]

        if not isinstance(self.model, nn.Sequential):
            raise ValueError("self.model must be a sequential self.model.")
        self.handles.append(self.model[0].register_forward_hook(fw_hook_get_input))
        for k in range(len(self.model)):
            self.handles.append(self.model[k].register_forward_hook(fw_hook_get_latent))

    def clean_up(self):
        for handle in self.handles:
            handle.remove()
            del handle
        gc.collect()

    def compute(self, loader, output_size):
        # keep track of running sum
        h = torch.zeros_like(parameters_to_vector(self.model.parameters()))

        for batch in loader:
            batch = [item.to(self.device) for item in batch]

            h += self.compute_batch(output_size, *batch)

        return h


class RmseHessianCalculator(HessianCalculator):
    def compute_batch(self, feature_maps):
        z = feature_maps[-1]
        bs, output_size = z.shape

        # Saves the product of the Jacobians wrt layer input
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=self.device))

        hessian = []
        with torch.no_grad():
            for k in range(len(self.model) - 1, -1, -1):
                if isinstance(self.model[k], torch.nn.Linear):
                    diag_elements = torch.einsum("bii->bi", tmp)
                    h_k = torch.einsum(
                        "bi,bj,bj->bij",
                        diag_elements,
                        feature_maps[k],
                        feature_maps[k],
                    )
                    h_k = h_k.view(bs, -1)
                    if self.model[k].bias is not None:
                        h_k = torch.cat([h_k, diag_elements], dim=1)

                    hessian = [h_k] + hessian

                if k == 0:
                    break

                # Calculate the Jacobian wrt to the inputs
                if isinstance(self.model[k], torch.nn.Linear):
                    jacobian_x = self.model[k].weight.expand((bs, *self.model[k].weight.shape))
                elif isinstance(self.model[k], torch.nn.Tanh):
                    jacobian_x = torch.diag_embed(
                        torch.ones(feature_maps[k + 1].shape, device=self.device) - feature_maps[k + 1] ** 2
                    )
                elif isinstance(self.model[k], torch.nn.ReLU):
                    jacobian_x = torch.diag_embed((feature_maps[k + 1] > 0).float())
                elif isinstance(self.model[k], L2Normalize):
                    jacobian_x = self.model[k]._jacobian_wrt_input(feature_maps[k], feature_maps[k + 1])
                else:
                    raise NotImplementedError

                # TODO: make more efficent by using row vectors
                tmp = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x, tmp, jacobian_x)

        return torch.cat(hessian, dim=1)


class FixedContrastiveHessianCalculator(HessianCalculator):
    def __init__(self, margin=0.2, alpha=0.01, num_classes=10, device="cpu") -> None:
        super().__init__(device)
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes
        self.rmse_hessian_calculator = RmseHessianCalculator(device)

        self.total_pairs = 0
        self.zeros = 0
        self.negatives = 0
    
    def init_model(self, model):
        super().init_model(model)
        self.rmse_hessian_calculator.init_model(model)

    def compute_batch(
        self,
        feature_maps1: List[Tensor],
        feature_maps2: List[Tensor],
        y: Tensor,
    ) -> Tensor:
        h1 = self.rmse_hessian_calculator.compute_batch(feature_maps1)
        h2 = self.rmse_hessian_calculator.compute_batch(feature_maps2)
        hessian = h1 + h2

        z1 = feature_maps1[-1]
        z2 = feature_maps2[-1]
        bs = z1.shape[0]

        non_match_mask = (1 - y).bool()
        square_norms: Tensor = torch.einsum("no,no->n", z1 - z2, z1 - z2)
        in_margin_mask = square_norms < self.margin

        zero_mask = torch.logical_and(non_match_mask, ~in_margin_mask)
        negative_mask = torch.logical_and(non_match_mask, in_margin_mask)
        
        self.zeros += zero_mask.sum().detach().item()
        self.negatives += negative_mask.sum().detach().item()
        self.total_pairs += bs

        # Set to zero for non-matches outside mask
        hessian[zero_mask] = 0

        # Set to negative for non-matches in mask, scale by 1/(n_classes-1)
        hessian[negative_mask] = -1 / (self.num_classes - 1) * hessian[negative_mask]

        return hessian.sum(dim=0)

    def compute_batch_pairs(self, hard_pairs) -> Tensor:
        ap, p, an, n = hard_pairs

        t = torch.cat(
            (
                torch.ones(p.shape[0], device=self.device),
                torch.zeros(n.shape[0], device=self.device),
            )
        ).to(self.device)

        feature_maps1 = [x[torch.cat((ap, an))] for x in self.feature_maps]
        feature_maps2 = [x[torch.cat((p, n))] for x in self.feature_maps]

        return self.compute_batch(feature_maps1, feature_maps2, t)


class ContrastiveHessianCalculator(HessianCalculator):
    def __init__(self, margin=0.2, alpha=0.01, num_classes=10, device="cpu", force_positive=False) -> None:
        super().__init__(device)
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes
        self.force_positive = force_positive

        self.total_pairs = 0
        self.zeros = 0
        self.negatives = 0

    def compute_batch(
        self, feature_maps1: List[Tensor], feature_maps2: List[Tensor], y: Tensor, *args, **kwargs
    ) -> Tensor:
        z1 = feature_maps1[-1]
        z2 = feature_maps2[-1]

        bs, output_size = z1.shape

        non_match_mask = (1 - y).bool()
        square_norms: Tensor = torch.einsum("no,no->n", z1 - z2, z1 - z2)
        in_margin_mask = square_norms < self.margin

        zero_mask = torch.logical_and(non_match_mask, ~in_margin_mask)
        negative_mask = torch.logical_and(non_match_mask, in_margin_mask)
        
        self.zeros += zero_mask.sum().detach().item()
        self.negatives += negative_mask.sum().detach().item()
        self.total_pairs += bs

        # Saves the product of the Jacobians wrt layer input
        tmp1 = torch.diag_embed((1 + self.alpha) * torch.ones(bs, output_size, device=self.device))
        tmp2 = torch.diag_embed((1 + self.alpha) * torch.ones(bs, output_size, device=self.device))
        tmp3 = torch.diag_embed(torch.ones(bs, output_size, device=self.device))

        hessian = []
        with torch.no_grad():
            for k in range(len(self.model) - 1, -1, -1):
                # Calculate Hessian for linear layers (since they have parameters)
                if isinstance(self.model[k], torch.nn.Linear):
                    diag_elements1 = torch.einsum("bii->bi", tmp1)
                    diag_elements2 = torch.einsum("bii->bi", tmp2)
                    diag_elements3 = torch.einsum("bii->bi", tmp3)

                    h1 = torch.einsum(
                        "bi,bj,bj->bij",
                        diag_elements1,
                        feature_maps1[k],
                        feature_maps1[k],
                    ).view(bs, -1)
                    h2 = torch.einsum(
                        "bi,bj,bj->bij",
                        diag_elements2,
                        feature_maps2[k],
                        feature_maps2[k],
                    ).view(bs, -1)
                    h3 = torch.einsum(
                        "bi,bj,bj->bij",
                        diag_elements3,
                        feature_maps1[k],
                        feature_maps2[k],
                    ).view(bs, -1)

                    if self.model[k].bias is not None:
                        h1 = torch.cat([h1, diag_elements1], dim=1)
                        h2 = torch.cat([h2, diag_elements2], dim=1)
                        h3 = torch.cat([h3, diag_elements3], dim=1)

                    h_k = h1 + h2 - 2 * h3

                    hessian = [h_k] + hessian

                if k == 0 or isinstance(self.model[k], torch.nn.Conv2d):
                    break

                # Calculate the Jacobian wrt to the inputs
                if isinstance(self.model[k], torch.nn.Linear):
                    jacobian_x1 = self.model[k].weight.expand((bs, *self.model[k].weight.shape))
                    jacobian_x2 = jacobian_x1
                elif isinstance(self.model[k], torch.nn.Tanh):
                    jacobian_x1 = torch.diag_embed(
                        torch.ones(feature_maps1[k + 1].shape, device=self.device) - feature_maps1[k + 1] ** 2
                    )
                    jacobian_x2 = torch.diag_embed(
                        torch.ones(feature_maps2[k + 1].shape, device=self.device) - feature_maps2[k + 1] ** 2
                    )
                elif isinstance(self.model[k], torch.nn.ReLU):
                    jacobian_x1 = torch.diag_embed((feature_maps1[k + 1] > 0).float())
                    jacobian_x2 = torch.diag_embed((feature_maps2[k + 1] > 0).float())
                elif isinstance(self.model[k], L2Normalize):
                    jacobian_x1 = self.model[k]._jacobian_wrt_input(feature_maps1[k], feature_maps1[k + 1])
                    jacobian_x2 = self.model[k]._jacobian_wrt_input(feature_maps2[k], feature_maps2[k + 1])
                else:
                    raise NotImplementedError

                # Calculate the product of the Jacobians
                # TODO: make more efficent by using row vectors
                # Right now this is 96-97% of our runtime
                tmp1 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x1, tmp1, jacobian_x1)
                tmp2 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x2, tmp2, jacobian_x2)
                tmp3 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x1, tmp3, jacobian_x2)
        hessian = torch.cat(hessian, dim=1)

        # Set to zero for non-matches outside mask
        hessian = torch.einsum("b,bm->bm", torch.where(zero_mask, 0., 1.), hessian)

        # Set to negative for non-matches in mask, scale by 1/(n_classes-1)
        hessian = torch.einsum(
            "b,bm->bm",
            torch.where(negative_mask, -1. / (self.num_classes - 1.), 1.),
            hessian,
        )

        # Scale by the number of samples

        hessian = hessian.sum(dim=0)

        if self.force_positive:
            hessian = torch.max(hessian, torch.zeros_like(hessian))

        return hessian

    def compute_batch_pairs(self, hard_pairs) -> Tensor:
        ap, p, an, n = hard_pairs

        t = torch.cat(
            (
                torch.ones(p.shape[0], device=self.device),
                torch.zeros(n.shape[0], device=self.device),
            )
        ).to(self.device)

        feature_maps1 = [x[torch.cat((ap, an))] for x in self.feature_maps]
        feature_maps2 = [x[torch.cat((p, n))] for x in self.feature_maps]

        return self.compute_batch(feature_maps1, feature_maps2, t)
