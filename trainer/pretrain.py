import contextlib
import os
from math import sqrt

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from utils import load_model

def im2tokens(x):
    B, C, Hp, Wp = x.shape
    x = x.view(B, C, Hp * Wp)
    x = x.permute(0, 2, 1)
    return x

def tokens2im(x, with_cls=False):
    if with_cls:
        x = x[:, 1:, :]
    B, D, C = x.shape
    x = x.permute(0, 2, 1)
    x = x.view(B, C, int(sqrt(D)), int(sqrt(D)))
    return x

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class PreTrainerModule(pl.LightningModule):
    def __init__(self, 
            model_name: str,
            model_init_args: dict, 
            lr: float = 1e-4, 
            weight_decay: float = 0.0):
        super().__init__()
        self.model = load_model(model_name, model_init_args)
        self.lr = lr
        self.weight_decay = weight_decay
        self.projector = self._get_projector()
        self.teacher = self._get_teacher()


    def _get_projector(self):
        if self.model.dim != 768:
            return nn.Linear(self.model.dim, 768, bias=False)
        else:
            return nn.Identity()

    def _get_teacher(self):
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            teacher_model = torch.hub.load(
                "amaralibey/bag-of-queries",
                "get_trained_boq",
                backbone_name="dinov2",
                output_dim=12288,
            )

        teacher_model.eval()
        freeze(teacher_model)
        return teacher_model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        s_img, t_img = batch
        T_features = im2tokens(self.teacher.backbone(t_img))
        S_features = self.projector(self(s_img))
        loss = self._feature_loss(T_features, S_features)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        s_img, t_img = batch
        T_features = im2tokens(self.teacher.backbone(t_img))
        S_features = self.projector(self(s_img))
        loss = self._feature_loss(T_features, S_features)
        self.log("val_loss", loss)

    def _interpolate_features(self, T_features, S_features):
        B, D, C = T_features.shape
        S_features = tokens2im(S_features, with_cls=True)
        T_features = tokens2im(T_features, with_cls=False)
        S_features = F.interpolate(
            S_features,
            size=(T_features.shape[2], T_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return T_features, S_features

    def _feature_loss(self, T_features, S_features):
        T_features, S_features = self._interpolate_features(T_features, S_features)
        loss = torch.nn.functional.mse_loss(T_features, S_features)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
