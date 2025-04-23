from collections import defaultdict

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import DataLoader
from PIL import Image
from datasets import ALL_DATASETS


def pair(t):
    return t if isinstance(t, (list, tuple)) else [t]




def match_cosine(global_desc, num_references, ground_truth, k_values=[1, 5, 10]):
    global_desc = global_desc.cpu().numpy()
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    index = faiss.IndexFlatIP(reference_desc.shape[1])
    index.add(reference_desc)

    dist, predictions = index.search(query_desc, max(k_values))

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break
    d = {}
    correct_at_k = (correct_at_k / len(predictions)) * 100
    for k, v in zip(k_values, correct_at_k):
        d[k] = v
    return d


class EvaluateModule(pl.LightningModule):
    def __init__(self, model: nn.Module, dataset_names: list[str], image_size: int, batch_size: int, num_workers: int, val_dataset_dir: str):
        super().__init__()
        self.model = model
        self.dataset_names = dataset_names
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_dataset_dir = val_dataset_dir

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
            for dataset_name in self.dataset_names:
                assert (
                    dataset_name.lower() in ALL_DATASETS.keys()
                ), f"Dataset {dataset_name} not found, must choose from {ALL_DATASETS.keys()}"

            self.test_datasets = [
                ALL_DATASETS[dataset_name](
                    val_dataset_dir=self.val_dataset_dir,
                    input_transform=self._transform(),
                    which_set="test",
                )
                for dataset_name in self.dataset_names
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
            self.test_descriptors[dataset.__repr__()] = torch.zeros((len(dataset), desc_dim))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.test_datasets[dataloader_idx].__repr__()
        images, indices = batch          # <-- keep the indices
        desc = self(images).detach().cpu()
        self.test_descriptors[dataset_name][indices] = desc

    def on_test_end(self):
        # Collect all results first
        all_recalls = {}
        ks = (1, 5, 10)
        
        for dataset in self.test_datasets:
            dataset_name = dataset.__repr__()
            descs = self.test_descriptors[dataset_name]
            gt = dataset.ground_truth
            num_references = dataset.num_references
            recalls = match_cosine(descs, num_references, gt, k_values=[1, 5, 10])
            print(recalls)
            all_recalls[dataset_name] = recalls

        # Create table with datasets as rows and k values as columns
        headers = ["Dataset"] + [f"R@{k}" for k in ks]
        table_data = []
        for dataset_name, recalls in all_recalls.items():
            row = [dataset_name] + [f"{recalls[k]:.1f}%" for k in ks]
            table_data.append(row)
            
        print("\nResults:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
