import torch


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


x = torch.randn(5, 5)
print(sparsify(x, 0.2))
