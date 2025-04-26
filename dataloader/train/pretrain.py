import os
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


def search_paths(dir: str, levels: int = 3):
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def safe_pretrain_collate_fn(batch):
    batch = [
        sample
        for sample in batch
        if sample is not None and sample[0] is not None and sample[1] is not None
    ]

    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    student_imgs, teacher_imgs = zip(*batch)
    student_batch = torch.stack(student_imgs, dim=0)
    teacher_batch = torch.stack(teacher_imgs, dim=0)
    return student_batch, teacher_batch


def filter_panorama_images(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width > height * 1.5:
        crop_width = height
        left = torch.randint(0, width - crop_width + 1, (1,)).item()
        img = img.crop((left, 0, left + crop_width, height))
    return img
    

class PretrainDataset(Dataset):
    def __init__(
        self,
        train_data_dir: str,
        student_transform: T.Compose,
        teacher_transform: T.Compose,
        split: str = "train",
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.split = split
        self.img_paths = self._data_split(search_paths(train_data_dir), split)
        assert self.split in ["train", "val"], "split must be either 'train' or 'val'"
        self._print_stats()

        
    def _print_stats(self):
        print("\n" + "="*50)
        print(f"Dataset Summary:")
        print(f"Root Directory: {self.train_data_dir}")
        print(f"Split: {self.split}")
        print(f"Number of Images: {len(self.img_paths)}")
        print("="*50 + "\n")

    def _data_split(self, img_paths: list, split: str):
        if split == "train":
            return img_paths[: int(len(img_paths) * 0.95)]
        else:
            return img_paths[int(len(img_paths) * 0.95) :]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = filter_panorama_images(img)
            s_img = self.student_transform(img)
            t_img = self.teacher_transform(img)
            return s_img, t_img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        img_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.teacher_transform = self._teacher_transform()
        self.train_transform = self._train_transform()
        self.val_transform = self._val_transform()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PretrainDataset(
                self.train_data_dir,
                self.train_transform,
                self.teacher_transform,
                split="train",
            )
            self.val_dataset = PretrainDataset(
                self.train_data_dir, self.val_transform, self.teacher_transform, split="val"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=safe_pretrain_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=safe_pretrain_collate_fn,
        )

    def _teacher_transform(self):
        return T.Compose(
            [
                T.Resize(
                    pair(322), interpolation=T.InterpolationMode.BICUBIC, antialias=True
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _train_transform(self):
        return T.Compose(
            [
                T.Resize(
                    pair(self.img_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _val_transform(self):
        return T.Compose(
            [
                T.Resize(
                    pair(self.img_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
