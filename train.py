import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import BoQModel
from trainer import EigenPlacesTrainer
from utils import load_config

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model = BoQModel(**config["model"])
    model_module = EigenPlacesTrainer(model=model, **config["model_module"])

    wandb_logger = WandbLogger(project="eigenplaces", dir="logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{lateral_loss:.2f}-{frontal_loss:.2f}",
        monitor="lateral_loss",
        mode="min",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        **config["trainer"], callbacks=[checkpoint_callback], logger=wandb_logger
    )
    trainer.fit(model_module)


if __name__ == "__main__":
    main()
