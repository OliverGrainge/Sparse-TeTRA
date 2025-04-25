import logging

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as tfm
from tabulate import tabulate
from torch.utils.data import DataLoader

import trainer.augmentations as augmentations
from model import *
from trainer import commons
from utils import import_model_cls

from .cosfase_loss import MarginCosineProduct
from .datasets import EigenPlacesDataset
from .datasets.test_dataset import TestDataset

RECALL_VALUES = [1, 5, 10, 20]


class VPRTrainer(pl.LightningModule):
    def __init__(
        self,
        # model parameters
        model_name: str,
        model_init_args: dict,
        # data parameters
        data_dir: str,
        val_dataset_dir: str,
        # dataset/grid sampling
        image_size: int,
        M: int,
        N: int,
        focal_dist: int,
        min_images_per_class: int,
        visualize_classes: bool,
        groups_num: int,
        # training hyperparameters
        batch_size: int,
        num_workers: int,
        iterations_per_epoch: int,
        fc_output_dim: int,
        s: float,
        m: float,
        classifiers_lr: float,
        model_lr: float,
        lambda_lat: float,
        lambda_front: float,
        # optional resume
        resume_model: str = None,
        brightness: float = 0.7,
        contrast: float = 0.7,
        saturation: float = 0.7,
        hue: float = 0.5,
        random_resized_crop: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # build model
        self.model = self._load_model()
        if self.hparams.resume_model:
            state = torch.load(self.hparams.resume_model)
            self.model.load_state_dict(state)

        # manual optimization
        self.automatic_optimization = False

        # loss & amp
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def _load_model(self):
        cls = import_model_cls(self.hparams.model_name)
        return cls(**self.hparams.model_init_args)

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, dim=1)
        return x

    def setup(self, stage=None):
        # prepare training groups
        total_groups = self.hparams.groups_num * 2
        self.groups = [
            EigenPlacesDataset(
                self.hparams.data_dir,
                M=self.hparams.M,
                N=self.hparams.N,
                focal_dist=self.hparams.focal_dist,
                current_group=i // 2,
                min_images_per_class=self.hparams.min_images_per_class,
                angle=[0, 90][i % 2],
                visualize_classes=self.hparams.visualize_classes,
            )
            for i in range(total_groups)
        ]

        self.classifiers = torch.nn.ModuleList(
            [
                MarginCosineProduct(
                    self.hparams.fc_output_dim,
                    len(self.groups[i]),
                    self.hparams.s,
                    self.hparams.m,
                )
                for i in range(total_groups)
            ]
        )
        # validation & test sets
        self.val_ds = TestDataset(self.hparams.val_dataset_dir)

        # augmentation on GPU
        self.gpu_augmentation = tfm.Compose(
            [
                augmentations.DeviceAgnosticColorJitter(
                    brightness=self.hparams.brightness,
                    contrast=self.hparams.contrast,
                    saturation=self.hparams.saturation,
                    hue=self.hparams.hue,
                ),
                augmentations.DeviceAgnosticRandomResizedCrop(
                    [self.hparams.image_size, self.hparams.image_size],
                    scale=[1 - self.hparams.random_resized_crop, 1],
                ),
                tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_augmentation = tfm.Compose(
            [
                tfm.Resize((self.hparams.image_size, self.hparams.image_size)),
                tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def on_train_epoch_start(self):
        # select current group pair
        idx = (self.current_epoch % self.hparams.groups_num) * 2
        self.current_dataset_num = idx

        # infinite dataloaders for lateral & frontal
        self.iterators = []
        for offset in (0, 1):
            ds = self.groups[idx + offset]
            loader = commons.InfiniteDataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
            )
            self.iterators.append(iter(loader))

        # reset epoch metrics
        self.lateral_loss = torchmetrics.MeanMetric()
        self.frontal_loss = torchmetrics.MeanMetric()

    def train_dataloader(self):
        return DataLoader(
            list(range(self.hparams.iterations_per_epoch)),
            batch_size=1,
            num_workers=0,
        )

    def training_step(self, batch, batch_idx):
        # manual optimization
        optimizers = self.optimizers()
        opt_model, *opt_classifiers = optimizers

        opt_model.zero_grad()

        for local_idx, cls_opt in enumerate(
            (
                opt_classifiers[self.current_dataset_num],
                opt_classifiers[self.current_dataset_num + 1],
            )
        ):
            cls_opt.zero_grad()

            imgs, targets, _ = next(self.iterators[local_idx])
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            aug_imgs = self.gpu_augmentation(imgs)
            feats = self.model(aug_imgs)
            logits = self.classifiers[self.current_dataset_num + local_idx](
                feats, targets
            )
            loss = self.criterion(logits, targets)
            multiplier = (
                self.hparams.lambda_lat if local_idx == 0 else self.hparams.lambda_front
            )
            loss = loss * multiplier

            # backward & step classifier
            loss.backward()
            cls_opt.step()
            # update metrics
            if local_idx == 0:
                self.lateral_loss.update(loss.detach().cpu())
            else:
                self.frontal_loss.update(loss.detach().cpu())

        # step model optimizer
        opt_model.step()

        # log
        self.log(
            "lateral_loss", self.lateral_loss.compute(), on_epoch=True, prog_bar=False
        )
        self.log(
            "frontal_loss", self.frontal_loss.compute(), on_epoch=True, prog_bar=False
        )

    def configure_optimizers(self):
        # model + classifier optimizers
        model_opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.model_lr)
        cls_opts = [
            torch.optim.Adam(cls.parameters(), lr=self.hparams.classifiers_lr)
            for cls in self.classifiers
        ]
        return [model_opt] + cls_opts

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, indices = batch
        images = self.val_augmentation(images)
        descriptors = self(images)  # forward pass
        descriptors = descriptors.detach().cpu()  # detach & move to CPU
        output = {"indices": indices, "descriptors": descriptors}
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # Get the outputs from the validation steps
        outputs = self.validation_outputs

        # Create a big array to hold all descriptors
        all_descriptors = torch.zeros(
            len(self.val_ds), self.hparams.fc_output_dim, dtype=torch.float32
        )

        # Fill it in batch by batch
        for out in outputs:
            idx = out["indices"]
            desc = out["descriptors"]
            all_descriptors[idx] = desc

        # Now split into database vs. queries
        db_num = self.val_ds.database_num
        db_descriptors = all_descriptors[:db_num].numpy()
        query_descriptors = all_descriptors[db_num:].numpy()

        # Build Faiss index & search
        index = faiss.IndexFlatL2(self.hparams.fc_output_dim)
        index.add(db_descriptors)
        _, predictions = index.search(query_descriptors, max(RECALL_VALUES))

        # Compute recalls
        positives = self.val_ds.get_positives()
        recalls = np.zeros(len(RECALL_VALUES))
        for q_idx, preds in enumerate(predictions):
            for i, k in enumerate(RECALL_VALUES):
                if np.any(np.in1d(preds[:k], positives[q_idx])):
                    recalls[i:] += 1
                    break
        recalls = recalls / self.val_ds.queries_num * 100

        # Create table data with R@k as columns
        headers = ["Metric"] + [f"R@{k}" for k in RECALL_VALUES]
        table_data = [
            ["Value"] + [f"{recalls[i]:.2f}%" for i in range(len(RECALL_VALUES))]
        ]

        # Print the table
        print("\nValidation Recalls:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Log the main recall metric
        for i, k in enumerate(RECALL_VALUES):
            self.log(f"val_recall@{k}", recalls[i], prog_bar=False)

        del self.validation_outputs
        del all_descriptors, db_descriptors, query_descriptors, index, predictions, recalls
        
        if isinstance(index, faiss.Index):          # release FAISS GPU buffers
             index.reset()                           # drops vectors
             del index
 
        torch.cuda.empty_cache()                    # flush caching allocator 
        torch.cuda.ipc_collect()  
