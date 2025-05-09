from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tabulate import tabulate
import torch.profiler
from torch.utils.data import DataLoader

from dataloader.val import ALL_DATASETS
from trainer.matching import match_cosine

def pair(t):
    return (
        (t, t)
        if isinstance(t, int)
        else tuple(t[:2]) if isinstance(t, (list, tuple)) else (t, t)
    )

class EvaluateModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        val_data_dir: str,
        val_set_names: list[str],
        image_size: int,
        batch_size: int,
        num_workers: int,
        sparsity: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.val_set_names = val_set_names
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_data_dir = val_data_dir
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _transform(self):
        return T.Compose(
            [
                T.Resize(pair(self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    

    def setup(self, stage: str):
        if stage == "test":
            for dataset_name in self.val_set_names:
                assert (
                    dataset_name.lower() in ALL_DATASETS.keys()
                ), f"Dataset {dataset_name} not found, must choose from {ALL_DATASETS.keys()}"

            self.test_datasets = [
                ALL_DATASETS[dataset_name](
                    val_dataset_dir=self.val_data_dir,
                    input_transform=self._transform(),
                    which_set="test",
                )
                for dataset_name in self.val_set_names
            ]

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.test_datasets:
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=torch.cuda.is_available(),
                )
            )
        return dataloaders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _test_inputs(self):
        img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
        img = Image.fromarray(img)
        img = self._transform()(img)
        return img[None, ...]

    def on_test_start(self):
        desc_dim = self(self._test_inputs().to(self.device)).shape[1]
        self.test_descriptors = {}
        for dataset in self.test_datasets:
            self.test_descriptors[dataset.__repr__()] = torch.zeros(
                (len(dataset), desc_dim), dtype=torch.float16
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.test_datasets[dataloader_idx].__repr__()
        images, indices = batch
        desc = self(images).detach().cpu()
        self.test_descriptors[dataset_name][indices] = desc.to(dtype=torch.float16)


    def on_test_end(self):
        all_recalls = {}
        ks = (1, 5, 10)
        for dataset in self.test_datasets:
            dataset_name = dataset.__repr__()
            descs = self.test_descriptors[dataset_name]
            gt = dataset.ground_truth
            num_references = dataset.num_references
            recalls = match_cosine(
                descs, num_references, gt, k_values=[1, 5, 10]
            )
            all_recalls[dataset_name] = recalls

        headers = ["Dataset"] + [f"R@{k}" for k in ks]
        table_data = []
        for dataset_name, recalls in all_recalls.items():
            row = [dataset_name] + [f"{recalls[k]:.1f}%" for k in ks]
            table_data.append(row)

        print("\nResults: ", self.model.__class__.__name__)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        self.test_results = all_recalls