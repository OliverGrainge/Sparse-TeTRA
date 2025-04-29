import argparse

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader.train import PostTrainDataModule
from model import ViT
from trainer import PostTrainerModule
from common import load_config, load_pretrain_checkpoint2model

torch.set_float32_matmul_precision("high")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    wandb_logger = WandbLogger(project="Sparse-TeTRA-posttrain")
    config_basename = args.config.split("/")[-1].split(".")[0]
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/posttrain/{config_basename}",
        monitor="val_recall",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=config_basename + "_{epoch}-{val_recall:.4f}",
    )

    data_module = PostTrainDataModule(**config["posttrain"]["data"])
    model_module = PostTrainerModule(**config["posttrain"]["module"])
    trainer = pl.Trainer(**config["posttrain"]["trainer"], logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    main()
