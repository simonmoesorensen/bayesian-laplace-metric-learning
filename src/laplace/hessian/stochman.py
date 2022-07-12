from abc import abstractmethod
import gc

import torch

import sys
import torch.nn.functional as F

import time


def diag_structure(curr_method):
    diag_inp_m = curr_method == "approx"
    diag_out_m = curr_method in ("approx", "exact")
    diag_inp_h = curr_method == "approx"
    diag_out_h = curr_method == "approx"
    return diag_inp_m, diag_out_m, diag_inp_h, diag_out_h


class HessianCalculator:
    def __init__(self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def __call__(self, net, *args, **kwargs):
        pass

    def init_model(self, model):

        self.feature_maps = []
        self.handles = []

        def fw_hook_get_latent(module, input, output):
            self.feature_maps.append(output.detach())

        def fw_hook_get_input(module, input, output):
            self.feature_maps = [input[0].detach()]

        self.handles.append(model[0].register_forward_hook(fw_hook_get_input))
        for k in range(len(model)):
            self.handles.append(model[k].register_forward_hook(fw_hook_get_latent))

    def clean_up(self):
        for handle in self.handles:
            handle.remove()
            del handle
        gc.collect()


class MseHessianCalculator(HessianCalculator):
    def __init__(self, method):
        super(MseHessianCalculator, self).__init__()

        self.method = method  # block, exact, approx, mix

    def __call__(self, net, *args, **kwargs):

        x = self.feature_maps[-1]

        if x.ndim == 4:
            bs, c, h, w = x.shape
            output_size = c * h * w
        else:
            bs, output_size = x.shape

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

                if self.method == "mix":
                    # prev_layer = net[k - 1] if k > 0 else None
                    # next_layer = net[k + 1] if k < len(net) - 1 else None
                    diag_inp_m, diag_out_m, diag_inp_h, diag_out_h = diag_structure(curr_method)
                    # curr_method, diag_out_h = swap_curr_method(curr_method, diag_out_h, prev_layer, net[k], next_layer)

                t = time.time()
                h_k = net[k]._jacobian_wrt_weight_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    diag_inp_m,
                    diag_out_m,
                )
                if h_k is not None:
                    H = [h_k.sum(dim=0)] + H
                t1 += time.time() - t

                if k == 0:
                    break

                t = time.time()
                tmp = net[k]._jacobian_wrt_input_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
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
