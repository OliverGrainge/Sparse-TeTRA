from math import sqrt
from typing import Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.blocks import ViTAttention, ViTFeedForward
from model.heads import BoQ
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
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
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


def tokens2img(x: torch.Tensor, with_cls: bool = True) -> torch.Tensor:
    b, n, c = x.shape
    if with_cls:
        x = x[:, 1:, :]
    x = x.permute(0, 2, 1)
    h, w = int(sqrt(n)), int(sqrt(n))
    x = x.reshape(b, c, h, w)
    return x


class BoQModel(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
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
        proj_channels: int = 512,
        num_queries: int = 32,
        num_layers: int = 2,
        row_dim: int = 32,
    ):
        super().__init__()

        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            embedding_norm=embedding_norm,
            embedding_linear=embedding_linear,
            attention_linear_layer=attention_linear_layer,
            attention_norm_layer=attention_norm_layer,
            feedforward_linear_layer=feedforward_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            feedforward_activation_layer=feedforward_activation_layer,
        )

        self.boq = BoQ(
            in_channels=dim,
            proj_channels=proj_channels,
            num_queries=num_queries,
            num_layers=num_layers,
            row_dim=row_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        x = tokens2img(x, with_cls=True)
        x = self.boq(x)[0]
        return x





class ViTCLSModel(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
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
    ):
        super().__init__()

        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            embedding_norm=embedding_norm,
            embedding_linear=embedding_linear,
            attention_linear_layer=attention_linear_layer,
            attention_norm_layer=attention_norm_layer,
            feedforward_linear_layer=feedforward_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            feedforward_activation_layer=feedforward_activation_layer,
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x[:, 0, :].view(-1, self.dim)





class ResNet(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[],
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        if "swsl" in model_name or "ssl" in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        else:
            if "resnext50" in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif "resnet50" in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif "101" in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif "152" in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif "34" in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif "18" in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif "wide_resnet50_2" in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError("Backbone architecture not recognized!")

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if "34" in model_name or "18" in model_name:
            out_channels = 512

        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch"""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class CosPlace(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def ResNet34CosPlace(out_dim):
    return nn.Sequential(
        ResNet(
            model_name="resnet34",
            pretrained=True,
            layers_to_freeze=2,
            layers_to_crop=[4],
        ),
        CosPlace(in_dim=256, out_dim=out_dim),
    )
