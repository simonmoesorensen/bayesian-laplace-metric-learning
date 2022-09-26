
from cProfile import label
from torchvision import transforms
import torchvision.datasets as d

import os, sys

sys.path.append("../")
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet,FashionMNISTLinearNet
from src.baselines.PFE.models import FashionMNIST_PFE
from src.utils import filter_state_dict
import torch
from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
from src.lightning_modules.PFELightningModule import PFELightningModule
from src.lightning_modules.PostHocLaplaceLightningModule import PostHocLaplaceLightningModule
from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from pytorch_metric_learning import losses, miners
from dotmap import DotMap
import numpy as np
from tqdm import tqdm

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

args = {"latent_dim": 3,
        "data_dir": "/work3/frwa/datasets/",
        "dataset": "FashionMNIST",
        "batch_size": 32,
        "num_workers": 8,
        "gpu_id": [0],
        "model": "Posthoc",
        "model_path": "outputs/Backbone/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/Model_Epoch_60_Time_2022-09-24T110808_checkpoint.pth",
        "random_seed": 42,
        "log_dir": "",
        "vis_dir": ""}

class AbstractActivationJacobian:
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        return jac_in * jac.reshape(jac.shape + (1,) * n)

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class AbstractJacobian:
    """Abstract class that will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    """

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        return self._jacobian_wrt_input_mult_left_vec(x, val, identity(x))

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val
    
    
class Identity(nn.Module):
    """Identity module that will return the same input as it receives."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = x

        if jacobian:
            xs = x.shape
            jac = (
                torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
                .repeat(xs[0], 1, 1)
                .reshape(xs[0], *xs[1:], *xs[1:])
            )
            return val, jac
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in


class Linear(AbstractJacobian, nn.Linear):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight, bias=None).movedim(-1, 1)

    def _jacobian_wrt_input_transpose_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight.T, bias=None).movedim(-1, 1)

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        return self.weight

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1 = x.shape
        c2 = val.shape[1]
        out_identity = torch.diag_embed(torch.ones(c2, device=x.device))
        jacobian = torch.einsum("bk,ij->bijk", x, out_identity).reshape(b, c2, c2 * c1)
        if self.bias is not None:
            jacobian = torch.cat([jacobian, out_identity.unsqueeze(0).expand(b, -1, -1)], dim=2)
        return jacobian

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jk->bmk", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jm->bm", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nk->bmk", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nm->bm", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bjk,bkq->biq", jacobian, tmp, jacobian)

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        tmp_diag = torch.diagonal(tmp, dim1=1, dim2=2)
        return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp_diag)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bj,bjq->biq", jacobian, tmp_diag, jacobian)

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1 = x.shape
        c2 = val.shape[1]

        Jt_tmp_J = torch.bmm(tmp_diag.unsqueeze(2), (x**2).unsqueeze(1)).view(b, c1 * c2)

        if self.bias is not None:
            Jt_tmp_J = torch.cat([Jt_tmp_J, tmp_diag], dim=1)

        return Jt_tmp_J
    
class Tanh(AbstractActivationJacobian, nn.Tanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 1.0 - val**2
        return jac

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjk->bmk", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjm->bm", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.diag_embed(self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp_diag))

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = jac.view(x.shape[0], -1)
        tmp = jac**2 * tmp_diag
        return tmp



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
                "b,bij->bij", 1 / (norm + 1e-6), torch.diag(torch.ones(d, device=x.device)).expand(b, d, d)
            )
            - out
        )

        return out

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_input(x, val)
        return torch.einsum("bij,bik,bkl->bjl", jacobian, tmp, jacobian)

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        else:
            raise NotImplementedError

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


import time

def diag_structure(curr_method):

    diag_inp_m = curr_method == "approx"
    diag_out_m = curr_method in ("approx", "exact")
    diag_inp_h = curr_method == "approx"
    diag_out_h = curr_method == "approx"

    return diag_inp_m, diag_out_m, diag_inp_h, diag_out_h


class HessianCalculator:
    def __init__(self):
        super(HessianCalculator, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def compute_batch(self, *args, **kwargs):
        pass

    def compute(self, loader, model, output_size):
        raise NotImplementedError


class MseHessianCalculator(HessianCalculator):
    def __init__(self, method):
        super(MseHessianCalculator, self).__init__()

        self.method = method  # block, exact, approx, mix

    def __call__(self, net, feature_maps, x, *args, **kwargs):
        
        if x.ndim == 4:
            bs, c, h, w = x.shape
            output_size = c * h * w
        else:
            bs, output_size = x.shape

        feature_maps = [x] + feature_maps

        # if we use diagonal approximation or first layer is flatten
        tmp = torch.ones(output_size, device=x.device)  # [HWC]
        if self.method in ("block", "exact"):
            tmp = torch.diag_embed(tmp).expand(bs, -1, -1)
        elif self.method in ("approx", "mix"):
            tmp = tmp.expand(bs, -1)
        else:
            raise NotImplementedError

        curr_method = "approx" if self.method == "mix" else self.method
        diag_inp_m, diag_out_m, diag_inp_h, diag_out_h = diag_structure(curr_method)

        t1 = 0
        t2 = 0
        H = []
        with torch.no_grad():
            for k in range(len(net) - 1, -1, -1):

                # jacobian w.r.t weight
                t = time.time()
                h_k = net[k]._jacobian_wrt_weight_sandwich(
                    feature_maps[k],
                    feature_maps[k + 1],
                    tmp,
                    diag_inp_m,
                    diag_out_m,
                )
                if h_k is not None:
                    H = [h_k.sum(dim=0)] + H
                t1 += time.time() - t

                # If we're in the last (first) layer, then skip the input jacobian
                if k == 0:
                    break

                # jacobian w.r.t input
                t = time.time()
                tmp = net[k]._jacobian_wrt_input_sandwich(
                    feature_maps[k],
                    feature_maps[k + 1],
                    tmp,
                    diag_inp_h,
                    diag_out_h,
                )
                t2 += time.time() - t

        if self.method == "block":
            H = [H_layer for H_layer in H]
        else:
            H = torch.cat(H, dim=0)

        return H


args = DotMap(args)

model = FashionMNISTConvNet(latent_dim=args.latent_dim)    
model.load_state_dict(torch.load(args.model_path))
model.cuda()

data_module = FashionMNISTDataModule(
    args.data_dir,
    args.batch_size,
    args.num_workers,
    npos=1,
    nneg=0,
)

test_dataloader = data_module.test_dataloader()

targets = []
images = []
tmps = []
labels = []
for batch in tqdm(test_dataloader):
    image = batch[0].cuda()
    labels.append(batch[1])
    with torch.inference_mode():
        tmp = model.conv(image)
        pred = model.linear(tmp)
    targets.append(pred.cpu())
    images.append(image.cpu())
    tmps.append(tmp.cpu())
    
targets = torch.cat(targets, dim=0)
images = torch.cat(images, dim=0)
tmps = torch.cat(tmps, dim=0)
labels = torch.cat(labels, dim=0)

# find a random positive point
classes = torch.unique(labels)
dict_ = {}
for c in classes:
    dict_[c.item()] = torch.where(labels == c)[0]
    
random_pos_idx = []
for l in labels:
    potential = dict_[l.item()]
    random_pos_idx.append(potential[torch.randint(0, len(potential), (1,))])

random_pos_idx = torch.cat(random_pos_idx, dim=0)

train_loader = DataLoader(TensorDataset(tmps, targets[random_pos_idx]), batch_size=args.batch_size, pin_memory=True)

# User-specified LA flavor
model_stochman = nn.Sequential(Linear(288, 128), Tanh(), Linear(128, 3), L2Norm())
model_stochman.cuda()
model_stochman.load_state_dict(model.linear.state_dict())

hessian_calculator = MseHessianCalculator("exact")

feature_maps = []
def fw_hook_get_latent(module, input, output):
    feature_maps.append(output.detach())

for k in range(len(model_stochman)):
    model_stochman[k].register_forward_hook(fw_hook_get_latent)

hessian = None
for X, y in tqdm(train_loader):
    X = X.cuda()
    with torch.inference_mode():
        feature_maps = []
        x_rec = model_stochman(X)
    h_s = hessian_calculator.__call__(model_stochman, feature_maps, x_rec)

    if hessian is None:
        hessian = h_s
    else:
        hessian += h_s

import pdb; pdb.set_trace()

torch.save(hessian, "hessian.pt")

plt.plot(hessian.cpu().numpy())
plt.savefig("tmp.png")