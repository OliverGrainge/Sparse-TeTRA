import argparse

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader.train import PretrainDataModule
from model import ViT
from trainer import PreTrainerModule
from utils import load_config

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--noob", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    wandb_logger = WandbLogger(project="Sparse-TeTRA-pretrain")
    config_basename = args.config.split("/")[-1].split(".")[0]
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/pretrain/",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=config_basename + "_{epoch}-{train_loss:.4f}",
        every_n_train_steps=2000,
    )

    data_module = PretrainDataModule(**config["pretrain"]["data"])
    model_module = PreTrainerModule(**config["pretrain"]["module"])
    trainer = pl.Trainer(**config["pretrain"]["trainer"], logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    main()
