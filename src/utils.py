import torch
import torch.nn as nn
from torch import Tensor, nn


def get_pairs(y):
    
    y = y.view(-1)
    
    ap = an = torch.where(y == 0)[0]
    p = torch.where(y == 1)[0]
    n = torch.where(y == -1)[0]
    
    num_neg_per_anchor = n.shape[0] // ap.shape[0]
    an = an.repeat_interleave(num_neg_per_anchor)
    
    num_pos_per_anchor = p.shape[0] // ap.shape[0]
    ap = ap.repeat_interleave(num_pos_per_anchor)
    
    return (ap, p, an, n)

def filter_state_dict(state_dict, remove="module."):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith(remove):
            new_state_dict[key[len(remove) :]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


class L2Norm(nn.Module):
    """L2 normalization layer"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, d = x.shape

        norm = torch.norm(x, p=2, dim=1)

        out = torch.einsum("bi,bj->bij", x, x)
        out = torch.einsum("b,bij->bij", 1 / (norm**3 + 1e-6), out)
        out = (
            torch.einsum(
                "b,bij->bij",
                1 / (norm + 1e-6),
                torch.diag(torch.ones(d, device=x.device)).expand(b, d, d),
            )
            - out
        )

        return out


def separate_batchnorm_params(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend(
                    [param for param in [*layer.parameters()] if param.requires_grad]
                )
            else:
                paras_wo_bn.extend(
                    [param for param in [*layer.parameters()] if param.requires_grad]
                )

    return paras_only_bn, paras_wo_bn


def load_model(model, dataset, embedding_size, model_path, **kwargs):

    if model == "DUL":
        from src.baselines.DUL.models import CIFAR10_DUL, MNIST_DUL, Casia_DUL

        if dataset == "MNIST":
            model = MNIST_DUL(embedding_size=embedding_size)
        elif dataset == "CIFAR10":
            model = CIFAR10_DUL(embedding_size=embedding_size)
        elif dataset == "CASIA":
            model = Casia_DUL(embedding_size=embedding_size)
        elif dataset == "FashionMNIST":
            model = MNIST_DUL(embedding_size=embedding_size)
    elif model == "HIB":
        from src.baselines.HIB.models import CIFAR10_HIB, MNIST_HIB, Casia_HIB

        if dataset == "MNIST":
            model = MNIST_HIB(embedding_size=embedding_size)
        elif dataset == "CIFAR10":
            model = CIFAR10_HIB(embedding_size=embedding_size)
        elif dataset == "CASIA":
            model = Casia_HIB(embedding_size=embedding_size)
        elif dataset == "FashionMNIST":
            model = MNIST_HIB(embedding_size=embedding_size)

        from src.baselines.PFE.models import (
            CIFAR10_PFE,
            MNIST_PFE,
            Casia_PFE,
            FashionMNIST_PFE,
        )

        if dataset == "MNIST":
            model = MNIST_PFE(embedding_size=embedding_size)
        elif dataset == "CIFAR10":
            model = CIFAR10_PFE(embedding_size=embedding_size)
        elif dataset == "CASIA":
            model = Casia_PFE(embedding_size=embedding_size)
        elif dataset == "FashionMNIST":
            model = FashionMNIST_PFE(kwargs["loss"], embedding_size=embedding_size)
    elif model == "Laplace":
        # if dataset == "MNIST":
        #     model = MNIST_Laplace(embedding_size=embedding_size)
        # elif dataset == "CIFAR10":
        #     model = CIFAR10_Laplace(embedding_size=embedding_size)
        # elif dataset == "CASIA":
        #     model = Casia_Laplace(embedding_size=embedding_size)
        pass
    else:
        raise ValueError(f"{model=} and {dataset=} not found or not supported")

    model.load_state_dict(torch.load(model_path))
    return model
