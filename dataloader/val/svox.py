import os
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class SVOX(Dataset):
    def __init__(
        self,
        val_dataset_dir=None,
        input_transform=None,
        which_set="test",
        condition=None,
    ):
        if which_set != "test":
            raise ValueError("SVOX only supports test set")

        assert condition in [
            "overcast",
            "sun",
            "snow",
            None,
            "rain",
            "night",
        ], f"SVOX only supports overcast, sun, snow, rain, and night. You gave {condition}"
        self.condition = condition
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir)
        self.which_set = which_set
        assert which_set == "test", "Tokyo247 only supports test set"
        # reference images names

        if condition is None:
            self.dbImages = np.load(
                f"dataloader/val/image_paths/svox_{which_set}_dbImages.npy"
            )
        else:
            self.dbImages = np.load(
                f"dataloader/val/image_paths/svox_{condition}_{which_set}_dbImages.npy"
            )

        # query images names
        if condition is None:
            self.qImages = np.load(f"dataloader/val/image_paths/svox_{which_set}_qImages.npy")
        else:
            self.qImages = np.load(
                f"dataloader/val/image_paths/svox_{condition}_{which_set}_qImages.npy"
            )

        # ground truth
        if condition is None:
            self.ground_truth = np.load(
                f"dataloader/val/image_paths/svox_{which_set}_gt.npy", allow_pickle=True
            )
        else:
            self.ground_truth = np.load(
                f"dataloader/val/image_paths/svox_{condition}_{which_set}_gt.npy",
                allow_pickle=True,
            )
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index])).convert(
            "RGB"
        )
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"SVOX-{self.condition}"


class SVOXNight(SVOX):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        super().__init__(val_dataset_dir, input_transform, which_set, "night")


class SVOXRain(SVOX):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        super().__init__(val_dataset_dir, input_transform, which_set, "rain")


class SVOXSun(SVOX):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        super().__init__(val_dataset_dir, input_transform, which_set, "sun")


class SVOXSnow(SVOX):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        super().__init__(val_dataset_dir, input_transform, which_set, "snow")


class SVOXOvercast(SVOX):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        super().__init__(val_dataset_dir, input_transform, which_set, "overcast")