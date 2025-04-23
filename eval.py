import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from model import *

from baselines import ALL_BASELINES, IMAGE_SIZES
from evaluation import EvaluateModule
from model import BoQModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=str,
        required=False,
        choices=["eigenplaces", "boq", "cosplace"],
        default="",
    )
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument(
        "--test_datasets", type=str, required=False, nargs="+", default=["pitts30k"]
    )
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--num_workers", type=int, required=False, default=16)
    parser.add_argument("--val_dataset_dir", type=str, required=False, default="/home/oliver/datasets_drive/vpr_datasets")
    return parser.parse_args()


def load_state_dict(model: nn.Module, checkpoint_path: str):
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return model


def main():
    args = parse_args()
    if len(args.baseline) > 0:
        assert (
            args.baseline in [k.lower() for k in ALL_BASELINES.keys()]
        ), f"Baseline {args.baseline} not found, must choose from {[k.lower() for k in ALL_BASELINES.keys()]}"
        model = ALL_BASELINES[args.baseline]()
        image_size = IMAGE_SIZES[args.baseline]
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        model = BoQModel(**config["model"])
        image_size = config["image_size"]
        model = load_state_dict(model, args.checkpoint)

    module = EvaluateModule(
        model=model, dataset_names=args.test_datasets, image_size=image_size, batch_size=args.batch_size, num_workers=args.num_workers, val_dataset_dir=args.val_dataset_dir
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
    )
    trainer.test(module)


if __name__ == "__main__":
    main()