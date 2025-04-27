from trainer import EvaluateModule, PostTrainerModule
import pytorch_lightning as pl
import argparse
from utils import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def main(): 
    args = parse_args()
    config = load_config(args.config)
    posttrain_module = PostTrainerModule.load_from_checkpoint(config["eval"]["model_module"]["checkpoint_path"])
    model = posttrain_module.model
    model.eval()

    eval_module = EvaluateModule(model, **config["eval"]["evaluate_module"])
    trainer = pl.Trainer(**config["eval"]["trainer"])
    trainer.test(eval_module)


if __name__ == "__main__":
    main()