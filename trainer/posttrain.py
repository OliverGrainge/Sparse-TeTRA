import contextlib
import os
from math import sqrt

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.optim import lr_scheduler
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from model.aggregation import BoQ, CLS, CosPlace, SALAD, MixVPR
from model.models import SparseTernaryViT
import random 

from trainer.matching import match_cosine
from tabulate import tabulate
from PIL import Image
import numpy as np
from utils import load_posttrain_checkpoint2model
from model.models import ViT 


def im2tokens(x):
    B, C, Hp, Wp = x.shape
    x = x.reshape(B, C, Hp * Wp)
    x = x.permute(0, 2, 1)
    return x

def tokens2im(x, with_cls=False):
    if with_cls:
        x = x[:, 1:, :]
    B, D, C = x.shape
    x = x.permute(0, 2, 1)
    x = x.reshape(B, C, int(sqrt(D)), int(sqrt(D)))   # <- SAFE NOW
    return x


def pair(t):
    return (
        (t, t)
        if isinstance(t, int)
        else tuple(t[:2]) if isinstance(t, (list, tuple)) else (t, t)
    )

    

def load_agg_method(agg_method: str): 
    if agg_method.lower() == 'cls': 
        return CLS
    elif agg_method.lower() == 'salad': 
        return SALAD
    elif agg_method.lower() == 'boq': 
        return BoQ
    elif agg_method.lower() == 'cosplace': 
        return CosPlace
    elif agg_method.lower() == 'mixvpr': 
        return MixVPR
    else: 
        raise ValueError(f"Aggregation method {agg_method} not found")
    

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class SparseModel(nn.Module): 
    def __init__(self, backbone, agg): 
        super().__init__()
        self.backbone = backbone
        self.agg = agg

    def forward(self, x, sparsity: float = 0.0): 
        return self.agg(self.backbone(x, sparsity))
        

class PostTrainerModule(pl.LightningModule):
    def __init__(self,
            checkpoint_path: str,
            val_sparsity: float,
            agg_name: str, 
            agg_init_kwargs: dict,
            
            #---- Train hyperparameters
            lr: float = 0.03, 
            optimizer: str = 'sgd',
            weight_decay: float = 1e-3,
            momentum: float = 0.9,
            warmpup_steps: int = 500,
            milestones: list[int] = [5, 10, 15],
            lr_mult: float = 0.3,
            ):
        super().__init__()
        self.model = self._setup_model(agg_name, agg_init_kwargs, checkpoint_path)
        self.val_sparsity = val_sparsity
        self.image_size = 224
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.save_hyperparameters(ignore=['model'])

    def _setup_model(self, agg_name, agg_init_kwargs, checkpoint_path): 
        backbone = SparseTernaryViT()
        #backbone = load_posttrain_checkpoint2model(backbone, checkpoint_path)
        agg_method = load_agg_method(agg_name)
        return SparseModel(backbone, agg_method(**agg_init_kwargs))

    def setup(self, stage: str): 
        if stage == 'fit': 
            self._miner = MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
            self._loss_fn = MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())


    def forward(self, x, sparsity: float = 0.0): 
        return self.model(x, sparsity)
    
    def _transform(self):
        return T.Compose(
            [
                T.Resize(pair(self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    def _test_inputs(self):
        img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
        img = Image.fromarray(img)
        img = self._transform()(img)
        return img[None, ...]
    
    def _loss_function(self, descriptors, labels): 
        miner_outputs = self._miner(descriptors, labels) 
        loss = self._loss_fn(descriptors, labels, miner_outputs)
        return loss 
    
    def training_step(self, batch, batch_idx): 
        places, labels = batch 
        BS, N, ch, h, w = places.shape 
        images = places.reshape(BS*N, ch, h, w) 
        labels = labels.reshape(-1) 
        images, labels = images.contiguous(), labels.contiguous()
        sparsity = random.uniform(0.1, 0.6)
        descriptors = self(images, sparsity) 
        loss = self._loss_function(descriptors, labels)
        self.log('train_loss', loss)
        return loss 

    def on_validation_epoch_start(self):
        desc_dim = self(self._test_inputs().to(self.device), self.val_sparsity).shape[1]
        self.test_descriptors = {}
        for dataset in self.trainer.datamodule.val_datasets:
            self.test_descriptors[dataset.__repr__()] = torch.zeros(
                (len(dataset), desc_dim), dtype=torch.float16
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.trainer.datamodule.val_datasets[dataloader_idx].__repr__()
        images, indices = batch
        desc = self(images, self.val_sparsity).detach().cpu()
        self.test_descriptors[dataset_name][indices] = desc.to(dtype=torch.float16)

    def on_validation_epoch_end(self):
        all_recalls = {}
        ks = (1, 5, 10)

        for dataset in self.trainer.datamodule.val_datasets:
            dataset_name = dataset.__repr__()
            descs = self.test_descriptors[dataset_name]
            gt = dataset.ground_truth
            num_references = dataset.num_references
            recalls = match_cosine(
                descs, num_references, gt, k_values=[1, 5, 10]
            )
            all_recalls[dataset_name] = recalls
            self.log(f'{dataset_name}_R1', recalls[1])
            self.log(f'{dataset_name}_R5', recalls[5])
            self.log(f'{dataset_name}_R10', recalls[10])

        headers = ["Dataset"] + [f"R@{k}" for k in ks]
        table_data = []
        for dataset_name, recalls in all_recalls.items():
            row = [dataset_name] + [f"{recalls[k]:.1f}%" for k in ks]
            table_data.append(row)

        print("\nResults: ", self.model.__class__.__name__)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        log_dataset = self.trainer.datamodule.val_datasets[0].__repr__()
        self.log('val_recall', all_recalls[log_dataset][1])

    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]