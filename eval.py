import argparse
import importlib

import pytorch_lightning as pl

from evaluation import EvaluateModule
from model import *
from model import BoQModel
from utils import load_config, load_lightning2model_checkpoint
import importlib

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def import_model_cls(baseline_name):
    """Case-insensitive search for function across candidate modules."""
    candidate_modules = ["model.baselines", "model.models"]
    
    for module_name in candidate_modules:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            if attr.lower() == baseline_name.lower():
                return getattr(module, attr)
    
    raise ImportError(
        f"Function '{baseline_name}' not found in any of {candidate_modules} (case-insensitive)"
    )

def main():
    args = parse_args()
    config = load_config(args.config)

    if "baseline" in config.keys():
        baseline_name = config["baseline"]["baseline_name"]
        model = import_model_cls(baseline_name)()
    else:
        model_cls = import_model_cls(baseline_name)
        model = model_cls(**config["model"])
        model = load_lightning2model_checkpoint(
            model, config["model"]["checkpoint_path"]
        )

    module = EvaluateModule(
        model=model,
        dataset_names=config["eval_module"]["val_datasets"],
        image_size=config["eval_module"]["image_size"],
        batch_size=config["eval_module"]["batch_size"],
        num_workers=config["eval_module"]["num_workers"],
        val_dataset_dir=config["eval_module"]["val_dataset_dir"],
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
    )
    trainer.test(module)


if __name__ == "__main__":
    main()
