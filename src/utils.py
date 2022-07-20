import torch
import torch.nn as nn


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
                paras_only_bn.extend([param for param in [*layer.parameters()] if param.requires_grad])
            else:
                paras_wo_bn.extend([param for param in [*layer.parameters()] if param.requires_grad])

    return paras_only_bn, paras_wo_bn
