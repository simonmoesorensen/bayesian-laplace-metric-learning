#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class JacType(Enum):
    """Class for declaring the type of an intermediate Jacobian.
    Options are:
        DIAG:   The Jacobian is a (B)x(N) matrix that represents
                a diagonal matrix of size (B)x(N)x(N)
        FULL:   The Jacobian is a matrix of whatever size."""

    DIAG = 1
    FULL = 2


class ActivationJacobian(ABC):
    """Abstract class for activation functions.

    Any activation functions subclassing this class will need to implement
    a method for computing their Jacobian."""

    def __abstract_init__(self, activation, *args, **kwargs):
        activation.__init__(self, *args, **kwargs)
        self.__activation__ = activation

    def forward(self, x, jacobian=False):
        val = self.__activation__.forward(self, x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    @abstractmethod
    def _jacobian(self, x, val):
        """Evaluate the Jacobian of an activation function.
        The Jacobian is evaluated at x, where the function
        attains value val."""
        pass

    def _jac_mul(self, x, val, Jseq, JseqType):
        """Multiply the Jacobian at x with M.
        This can potentially be done more efficiently than
        first computing the Jacobian, and then performing the
        multiplication."""

        J, _ = self._jacobian(x, val)  # (B)x(in) -- the current Jacobian;
        # should be interpreted as a diagonal matrix (B)x(in)x(in)

        # We either want to (matrix) multiply the current diagonal Jacobian with a
        #  *) vector: size (B)x(in)
        #       This should be interpreted as a diagonal matrix of size
        #       (B)x(in)x(in), and we want to perform the product diag(J) * diag(Jseq)
        #  *) matrix: size (B)x(in)x(M)
        #       In this case we want to return diag(J) * Jseq

        if JseqType is JacType.FULL:  # Jseq.dim() is 3: # Jseq is matrix
            Jseq = torch.einsum("bi,bim->bim", J, Jseq)  # diag(J) * Jseq
            # (B)x(in) * (B)x(in)x(M)-> (B)x(in)x(M)
            jac_type = JacType.FULL
        elif JseqType is JacType.DIAG:  # Jseq.dim() is 2: # Jseq is vector (representing a diagonal matrix)
            Jseq = J * Jseq  # diag(J) * diag(Jseq)
            # (B)x(in) * (B)x(in) -> (B)x(in)
            jac_type = JacType.DIAG
        else:
            print("ActivationJacobian:_jac_mul: What the hell?")
        return Jseq, jac_type


def __jac_mul_generic__(J, Jseq, JseqType):
    #
    # J: (B)x(K)x(in) -- the current Jacobian

    # We either want to (matrix) multiply the current Jacobian with a
    #  *) vector: size (B)x(in)
    #       This should be interpreted as a diagonal matrix of size
    #       (B)x(in)x(in), and we want to perform the product diag(J) * diag(Jseq)
    #  *) matrix: size (B)x(in)x(M)
    #       In this case we want to return diag(J) * Jseq
    if JseqType is JacType.FULL:  # Jseq.dim() is 3: # Jseq is matrix
        Jseq = torch.einsum("bki,bim->bkm", J, Jseq)  # J * Jseq
        # (B)x(K)(in) * (B)x(in)x(M)-> (B)x(K)x(M)
    elif JseqType is JacType.DIAG:  # Jseq.dim() is 2: # Jseq is vector (representing a diagonal matrix)
        Jseq = torch.einsum("bki,bi->bki", J, Jseq)  # J * diag(Jseq)
        # (B)x(K)(in) * (B)x(in) -> (B)x(K)x(in)
    else:
        print("__jac_mul_generic__: What the hell?")
    return Jseq, JacType.FULL


def __jac_add_generic__(J1, J1Type, J2, J2Type):
    # Add two Jacobians of possibly different types
    if J1Type is J2Type:
        J = J1 + J2
        JType = J1Type
    elif J1Type is JacType.FULL and J2Type is JacType.DIAG:
        J = J1 + torch.diag_embed(J2)
        JType = JacType.FULL
    elif J1Type is JacType.DIAG and J2Type is JacType.FULL:
        J = torch.diag_embed(J1) + J2
        JType = JacType.FULL
    else:
        print("__jac_add_generic__: What the ....?")

    return J, JType


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, jacobian=False, return_jac_type=False):
        xsh = x.shape
        if len(xsh) == 1:
            x = x.unsqueeze(0)

        x = x.view(-1, xsh[-1])  # Nx(d)

        if jacobian:
            Jseq = None

            for module in self._modules.values():
                val = module(x)
                if Jseq is None:
                    Jseq, JseqType = module._jacobian(x, val)
                else:
                    Jseq, JseqType = module._jac_mul(x, val, Jseq, JseqType)
                x = val
            x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))
            if JseqType is JacType.DIAG:
                Jseq = Jseq.view(xsh[:-1] + Jseq.shape[-1:])
            else:
                Jseq = Jseq.view(xsh[:-1] + Jseq.shape[-2:])

            if return_jac_type:
                return x, Jseq, JseqType
            else:
                return x, Jseq
        else:
            for module in self._modules.values():
                x = module(x)
            x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))

            return x

    def _jacobian(self, x, val):
        _, J, JType = self.forward(x, jacobian=True, return_jac_type=True)
        return J, JType

    def _jac_mul(self, x, val, Jseq, JseqType):
        for module in self._modules.values():
            val = module(x)
            Jseq, JseqType = module._jac_mul(x, val, Jseq, JseqType)
            x = val
        return Jseq, JseqType

    def inverse(self):
        layers = [L.inverse() for L in reversed(self._modules.values())]
        return Sequential(*layers)

    def dimensions(self):
        in_features, out_features = None, None
        for module in self._modules.values():
            if in_features is None and hasattr(module, "__constants__") and "in_features" in module.__constants__:
                in_features = module.in_features
            if hasattr(module, "__constants__") and "out_features" in module.__constants__:
                out_features = module.out_features
        return in_features, out_features

    def disable_training(self):
        state = []
        for module in self._modules.values():
            state.append(module.training)
            module.training = False
        return state

    def enable_training(self, state=None):
        if state is None:
            for module in self._modules.values():
                module.training = True
        else:
            for module, new_state in zip(self._modules.values(), state):
                module.training = new_state


class Reciprocal(nn.Module, ActivationJacobian):
    def __init__(self, b=0.0):
        super().__init__()
        self.b = b

    def forward(self, x, jacobian=False):
        val = 1.0 / (x + self.b)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = -((val) ** 2)
        return J, JacType.DIAG


class Sqrt(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x, jacobian=False):
        val = torch.sqrt(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = -0.5 / val
        return J, JacType.DIAG


class Norm2(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, jacobian=False):
        val = torch.sum(x**2, dim=self.dim, keepdim=True)

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = 2.0 * x.unsqueeze(1)
        return J, JacType.FULL

    def _jac_mul(self, x, val, Jseq, JseqType):
        J, _ = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return __jac_mul_generic__(J, Jseq, JseqType)


def fd_jacobian(function, x, h=1e-4):
    """Compute finite difference Jacobian of given function
    at a single location x. This function is mainly considered
    useful for debugging."""

    no_batch = x.dim() == 1
    if no_batch:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise Exception("The input should be a D-vector or a BxD matrix")
    B, D = x.shape

    # Compute finite differences
    E = h * torch.eye(D)
    try:
        # Disable "training" in the function (relevant eg. for batch normalization)
        orig_state = function.disable_training()
        Jnum = torch.cat([((function(x[b] + E) - function(x[b].unsqueeze(0))).t() / h).unsqueeze(0) for b in range(B)])
    finally:
        function.enable_training(orig_state)  # re-enable training

    if no_batch:
        Jnum = Jnum.squeeze(0)

    return Jnum


def jacobian_check(function, in_dim=None, h=1e-4, verbose=True):
    """Accepts an nnj module and checks the
    Jacobian via the finite differences method.

    Args:
        function:   An nnj module object. The
                    function to be tested.

    Returns a tuple of the following form:
    (Jacobian_analytical, Jacobian_finite_differences)
    """

    with torch.no_grad():
        batch_size = 5
        if in_dim is None:
            in_dim, _ = functions.dimensions()
            if in_dim is None:
                in_dim = 10
        x = torch.randn(batch_size, in_dim)
        try:
            orig_state = function.disable_training()
            y, J, Jtype = function(x, jacobian=True, return_jac_type=True)
        finally:
            function.enable_training(orig_state)

        if Jtype is JacType.DIAG:
            J = J.diag_embed()

        Jnum = fd_jacobian(function, x)

        if verbose:
            residual = (J - Jnum).abs().max()
            if residual > 100 * h:
                print(
                    "****** Warning: exceedingly large error:",
                    residual.item(),
                    "******",
                )
            else:
                print("OK (residual = ", residual.item(), ")")
        else:
            return J, Jnum


class L2Normalize(nn.Module):
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
            torch.einsum("b,bij->bij", 1 / (norm + 1e-6), torch.diag(torch.ones(d, device=x.device)).expand(b, d, d))
            - out
        )

        return out
