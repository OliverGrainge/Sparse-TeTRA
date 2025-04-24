import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import *
from trainer import VPRTrainer
from utils import import_model_cls, load_config, parse_args

torch.set_float32_matmul_precision("high")


def main():
    args = parse_args()
    config = load_config(args.config)
    config_name = os.path.basename(args.config).split(".")[0]
    model_module = VPRTrainer(**config["model"])
    wandb_logger = WandbLogger(project="eigenplaces", dir="logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{config_name}-" + "{epoch}-{lateral_loss:.2f}-{frontal_loss:.2f}",
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
