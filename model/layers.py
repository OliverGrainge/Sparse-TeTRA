from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class WeightQuantizer(Function):
    @staticmethod
    def forward(ctx, w):
        delta = 1e-5 + w.abs().mean()
        scale = 1.0 / delta
        qw = (w * scale).round().clamp_(-1, 1)
        dqw = delta * qw
        ctx.save_for_backward(delta)
        return dqw

    @staticmethod
    def backward(ctx, grad_output):
        (delta,) = ctx.saved_tensors
        return grad_output.clone()


def quant(x: torch.Tensor):
    return WeightQuantizer.apply(x)


class TernaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant(self.weight)
        output = F.linear(x, dqw, bias=self.bias)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


# =============================== Sparse-TeTRA ===============================


def sparsify(x: torch.Tensor, sparsity: float):
    with torch.no_grad():
        flat = x.abs().reshape(-1)
        k = int((1.0 - sparsity) * flat.numel())
        if k >= flat.numel():
            return x
        topk_idx = flat.topk(k, sorted=False).indices
        mask = torch.zeros_like(flat, dtype=x.dtype)
        mask[topk_idx] = 1.0
        mask = mask.view_as(x)
    mask = mask.detach()
    y = x * mask + (1.0 - mask) * (x - x.detach())
    return y


class SparseTernaryLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.4,
    ):
        super().__init__(in_features, out_features, bias)
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = sparsify(x, self.sparsity)
        dqw = quant(self.weight)
        x = F.linear(x, dqw, bias=self.bias)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
