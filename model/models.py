from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.blocks import ViTAttention, ViTFeedForward
from model.layers import SparseTernaryLinear, TernaryLinear

LAYERS_REGISTRY = {
    "linear": nn.Linear,
    "ternarylinear": TernaryLinear,
    "layernorm": nn.LayerNorm,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "softmax": nn.Softmax,
    "identity": nn.Identity,
    "sparseternarylinear": SparseTernaryLinear,
}


def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        feedforward_norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        feedforward_activation_layer: Type[nn.Module] = nn.GELU,
        attention_linear_layer: Type[nn.Linear] = nn.Linear,
        feedforward_linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ViTAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            norm_layer=attention_norm_layer,
                            linear_layer=attention_linear_layer,
                        ),
                        ViTFeedForward(
                            dim,
                            mlp_dim,
                            dropout=dropout,
                            norm_layer=feedforward_norm_layer,
                            activation_layer=feedforward_activation_layer,
                            linear_layer=feedforward_linear_layer,
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        in_channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0,
        embedding_norm: str = "LayerNorm",
        embedding_linear: str = "Linear",
        attention_linear_layer: str = "Linear",
        attention_norm_layer: str = "LayerNorm",
        feedforward_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        feedforward_activation_layer: str = "GELU",
    ) -> None:
        super().__init__()
        self.dim = dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_norm_layer=LAYERS_REGISTRY[attention_norm_layer.lower()],
            feedforward_norm_layer=LAYERS_REGISTRY[feedforward_norm_layer.lower()],
            feedforward_activation_layer=LAYERS_REGISTRY[
                feedforward_activation_layer.lower()
            ],
            attention_linear_layer=LAYERS_REGISTRY[attention_linear_layer.lower()],
            feedforward_linear_layer=LAYERS_REGISTRY[feedforward_linear_layer.lower()],
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
