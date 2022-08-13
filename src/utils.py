import torch
import torch.nn as nn
from pathlib import Path

# Create logs folder if not exists
root = Path(__file__).parent.parent
logs = root / "logs"

models = ["DUL", "HIB", "Backbone", "PFE"]
datasets = ["MNIST", "CIFAR10", "CASIA", "FashionMNIST"]

for model in models:
    for dataset in datasets:
        path = logs / model / dataset
        path.mkdir(parents=True, exist_ok=True)


class L2Norm(nn.Module):
    def forward(self, X):
        return l2_norm(X, axis=1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


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
    elif model == "PFE":
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
            model = FashionMNIST_PFE(kwargs['loss'], embedding_size=embedding_size)
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
