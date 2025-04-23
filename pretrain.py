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
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    wandb_logger = WandbLogger(project="logs/Sparse-TeTRA-pretrain")
    filename = f"vit_L[{config['model']['feedforward_linear_layer']}]_D[{config['model']['dim']}]"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{filename}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="{epoch}-{val_loss:.4f}",
    )

    data_module = PretrainDataModule(**config["data"])
    model = ViT(**config["model"])
    model_module = PreTrainerModule(model=model, **config["model_module"])
    trainer = pl.Trainer(**config["trainer"], logger=wandb_logger)
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    main()
