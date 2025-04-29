from trainer import EvaluateModule, PostTrainerModule
import pytorch_lightning as pl
import argparse
from utils import load_config
from model.baselines import DinoBoQ
from functools import partial
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def _load_baseline(model_name): 
    if model_name.lower() == "dinoboq": 
        return DinoBoQ()
    else: 
        raise ValueError(f"Model {model_name} not found")


def _load_model(config): 
    if "baseline" in config["eval"]["model_module"]: 
        model = _load_baseline(config["eval"]["model_module"]["baseline"])
        model.eval()
        return model 
    else: 
        posttrain_module = PostTrainerModule.load_from_checkpoint(config["eval"]["model_module"]["checkpoint_path"])
        sparsity = config["eval"]["model_module"]["sparsity"]
        model = posttrain_module.model
        model.forward = partial(model.forward, sparsity=config["eval"]["model_module"]["sparsity"])
        model.eval()
        return model

def add_baseline_result(config, module, result):
    """Add baseline evaluation results to CSV file
    
    Args:
        config: Config dict containing model info
        module: EvaluateModule instance
        result: Dict mapping dataset names to recall results
            Format: {
                'dataset_name': {
                    1: recall@1_value,
                    5: recall@5_value, 
                    10: recall@10_value
                }
            }
    """
    import os
    import pandas as pd
    
    if not os.path.exists("results/baselines.csv"):
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(columns=["model", "dataset", "recall@1", "recall@5", "recall@10"])
        df.to_csv("results/baselines.csv", index=False)
        
    df = pd.read_csv("results/baselines.csv")
    model_name = module.model.__class__.__name__
    
    for dataset, recalls in result.items():
        new_row = {
            "model": model_name,
            "dataset": dataset,
            "recall@1": recalls[1],
            "recall@5": recalls[5], 
            "recall@10": recalls[10]
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv("results/baselines.csv", index=False)

def add_posttrain_result(config, result): 
    pass 

def add_results(config, module, results):
    if "baseline" in config["eval"]["model_module"]: 
        add_baseline_result(config, module, results)
    else: 
        add_posttrain_result(config, results)

def main(): 
    args = parse_args()
    config = load_config(args.config)
    model = _load_model(config)

    eval_module = EvaluateModule(model, **config["eval"]["evaluate_module"])
    trainer = pl.Trainer(**config["eval"]["trainer"])
    trainer.test(eval_module)
    results = eval_module.test_results 
    print(results)


if __name__ == "__main__":
    main()