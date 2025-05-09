from typing import Optional, Type

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from model.layers import SparseTernaryLinear 

class ViTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        linear_layer: Type[nn.Module] = nn.Linear,
        linear_kwargs: dict = {},
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = norm_layer(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = linear_layer(dim, inner_dim * 3, bias=False, **linear_kwargs)

        self.to_out = (
            nn.Sequential(linear_layer(inner_dim, dim, **linear_kwargs), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ViTFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        linear_layer: Type[nn.Linear] = nn.Linear,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        activation_layer: Type[nn.Module] = nn.GELU,
        linear_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            norm_layer(dim),
            linear_layer(dim, hidden_dim, **linear_kwargs),
            activation_layer(),
            nn.Dropout(dropout),
            linear_layer(hidden_dim, dim, **linear_kwargs),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class SparseTernaryViTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = SparseTernaryLinear(dim, inner_dim * 3, bias=False)

        self.to_out = SparseTernaryLinear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x, sparsity).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out, sparsity)
        out = self.dropout(out)
        return out


class SparseTernaryViTFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.lin1 = SparseTernaryLinear(dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.lin2 = SparseTernaryLinear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = self.ln1(x) 
        x = self.lin1(x, sparsity) 
        x = self.act1(x)
        x = self.drop1(x)
        x = self.ln2(x)
        x = self.lin2(x, sparsity)
        x = self.drop2(x)
        return x
