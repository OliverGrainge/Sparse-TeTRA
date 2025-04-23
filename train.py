from model import BoQModel
from trainer import EigenPlacesTrainer
from utils import load_config 
import pytorch_lightning as pl 
import argparse 
import torch 

torch.set_float32_matmul_precision('high')

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()

def main(): 
    args = parse_args()
    config = load_config(args.config)
    model = BoQModel(**config["model"])
    model_module = EigenPlacesTrainer(
        model=model, 
        **config["model_module"]
        )
    trainer = pl.Trainer(**config["trainer"])
    trainer.fit(model_module)


if __name__ == "__main__":
    main()