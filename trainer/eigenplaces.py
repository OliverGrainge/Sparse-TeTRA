from collections import defaultdict
from typing import Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from tabulate import tabulate
from torch.nn import Parameter

from model import *
from utils import import_model_cls

from .datasets import EigenPlacesDataset


def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def cosine_sim(
    x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )


def get_iterator(
    groups, classifiers, classifiers_optimizers, batch_size, g_num, device, num_workers
):
    assert len(groups) == len(classifiers) == len(classifiers_optimizers)
    classifiers[g_num] = classifiers[g_num].to(device)
    move_to_device(classifiers_optimizers[g_num], device)
    return InfiniteDataLoader(
        groups[g_num],
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )


class VPRTrainer(pl.LightningModule):
    def __init__(
        self,
        # model parameters
        model_name: str,
        model_init_args: dict,
        # training parameters
        data_dir: str,
        val_dataset_dir: str,
        val_datasets: list[str],
        image_size: int,
        M: int,
        N: int,
        focal_dist: int,
        min_images_per_class: int,
        visualize_classes: int,
        groups_num: int,
        batch_size: int,
        num_workers: int,
        fc_output_dim: int,
        s: float,
        m: float,
        classifiers_lr: float,
        model_lr: float,
        lambda_lat: float,
        lambda_front: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._load_model()
        self.automatic_optimization = False

    def _load_model(self):
        model_cls = import_model_cls(self.hparams.model_name)
        model = model_cls(**self.hparams.model_init_args)
        return model

    def _val_transform(self):
        return T.Compose(
            [
                T.Resize(self.hparams.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.groups = [
                EigenPlacesDataset(
                    self.hparams.data_dir,
                    M=self.hparams.M,
                    N=self.hparams.N,
                    focal_dist=self.hparams.focal_dist,
                    current_group=n // 2,
                    min_images_per_class=self.hparams.min_images_per_class,
                    angle=[0, 90][n % 2],
                    visualize_classes=self.hparams.visualize_classes,
                    image_size=self.hparams.image_size,
                )
                for n in range(self.hparams.groups_num * 2)
            ]

            self.classifiers = [
                MarginCosineProduct(
                    self.hparams.fc_output_dim,
                    len(group),
                    s=self.hparams.s,
                    m=self.hparams.m,
                )
                for group in self.groups
            ]

            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_dataloader(self):
        self.current_dataset_num = (
            self.trainer.current_epoch % self.hparams.groups_num
        ) * 2
        opt = self.optimizers()
        classifiers_optimizers = opt[:-1]
        dataloaders = {
            "lateral": get_iterator(
                self.groups,
                self.classifiers,
                classifiers_optimizers,
                self.hparams.batch_size,
                self.current_dataset_num,
                self.device,
                self.hparams.num_workers,
            ),
            "frontal": get_iterator(
                self.groups,
                self.classifiers,
                classifiers_optimizers,
                self.hparams.batch_size,
                self.current_dataset_num + 1,
                self.device,
                self.hparams.num_workers,
            ),
        }
        return CombinedLoader(dataloaders, mode="min_size")

    def on_train_epoch_start(self):
        self.current_dataset_num = (
            self.trainer.current_epoch % self.hparams.groups_num
        ) * 2

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        classifiers_optimizers = opt[:-1]
        model_optimizer = opt[-1]
        model_optimizer.zero_grad()

        # Lateral Loss
        classifiers_optimizers[self.current_dataset_num].zero_grad()
        images, targets, _ = batch["lateral"]
        descriptors = self(images)
        output = (
            self.classifiers[self.current_dataset_num](descriptors, targets)
            * self.hparams.lambda_lat
        )
        loss = self.criterion(output, targets)
        self.manual_backward(loss)
        self.log("lateral_loss", loss)
        classifiers_optimizers[self.current_dataset_num].step()

        # Frontal Loss
        classifiers_optimizers[self.current_dataset_num + 1].zero_grad()
        images, targets, _ = batch["frontal"]
        descriptors = self(images)
        output = self.classifiers[self.current_dataset_num + 1](descriptors, targets)
        loss = self.criterion(output, targets) * self.hparams.lambda_front
        self.manual_backward(loss)
        self.log("frontal_loss", loss)
        classifiers_optimizers[self.current_dataset_num + 1].step()

        model_optimizer.step()

    def configure_optimizers(self):
        classifiers_optimizers = [
            torch.optim.Adam(classifier.parameters(), lr=self.hparams.classifiers_lr)
            for classifier in self.classifiers
        ]
        model_optimizer = [
            torch.optim.Adam(self.model.parameters(), lr=self.hparams.model_lr)
        ]
        return classifiers_optimizers + model_optimizer
