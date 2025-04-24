import argparse
import importlib
import os
from pathlib import Path
from typing import Type, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from evaluation import EvaluateModule
from model import *
from trainer import VPRTrainer
from utils import (import_model_cls, load_config,
                   load_lightning2model_checkpoint, parse_args)

# Constants
torch.set_float32_matmul_precision("high")


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def main() -> None:
    """Main function to run model evaluation."""
    args = parse_args()
    config = load_config(args.config)

    if "baseline" in config:
        baseline_name = config["baseline"]["baseline_name"]
        model = import_model_cls(baseline_name)()
    else:
        assert os.path.exists(
            config["model"]["checkpoint_path"]
        ), f"Checkpoint path {config['model']['checkpoint_path']} does not exist"
        model_module = VPRTrainer.load_from_checkpoint(
            config["model"]["checkpoint_path"]
        )
        model = model_module.model
        freeze(model)

    module = EvaluateModule(
        model=model,
        **config["eval_module"]
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
    )
    trainer.test(module)


if __name__ == "__main__":
    main()
