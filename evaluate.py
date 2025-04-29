from trainer import EvaluateModule, PostTrainerModule
import pytorch_lightning as pl
import argparse
import yaml 
from typing import Any, Dict
from model.baselines import DinoBoQ, DinoSalad
from functools import partial
import os
import pandas as pd
import sys
sys.path.append('/home/oliver/.cache/torch/hub/serizba_salad_main')
import os

from common import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--sparsity", type=float, default=0.0)
    return parser.parse_args()


def _load_baseline(model_name): 
    if model_name.lower() == "dinoboq": 
        return DinoBoQ()
    elif model_name.lower() == "dinosalad": 
        return DinoSalad()
    else: 
        raise ValueError(f"Model {model_name} not found")


def _load_model(config, args): 
    if "baseline" in config["eval"]: 
        model = _load_baseline(config["eval"]["baseline"])
        return model 
    else: 
        posttrain_module = PostTrainerModule.load_from_checkpoint(config["eval"]["checkpoint_path"])
        sparsity = args.sparsity
        model = posttrain_module.model
        model.forward = partial(model.forward, sparsity=sparsity)
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
    if not os.path.exists("results/baselines.csv"):
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(columns=["model", "dataset", "recall@1", "recall@5", "recall@10", "flops"])
        df.to_csv("results/baselines.csv", index=False)
        
    df = pd.read_csv("results/baselines.csv")
    model_name = module.model.__repr__()
    
    flops = result.pop("flops")
    for dataset, recalls in result.items():
        # Check if combination already exists
        if not df[(df['model'] == model_name) & (df['dataset'] == dataset)]:
            continue
            
        new_row = {
            "model": model_name,
            "dataset": dataset,
            "recall@1": recalls[1],
            "recall@5": recalls[5], 
            "recall@10": recalls[10],
            "flops": flops,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv("results/baselines.csv", index=False)

def add_posttrain_result(config, module, result): 
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
    if not os.path.exists("results/results.csv"):
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(columns=["model", "dataset", "recall@1", "recall@5", "recall@10", "flops", "sparsity"])
        df.to_csv("results/results.csv", index=False)
        
    df = pd.read_csv("results/results.csv")
    model_name = module.model.__repr__()
    
    flops = result.pop("flops")
    sparsity = result.pop("sparsity")
    for dataset, recalls in result.items():
        # Check if combination already exists
        if len(df[(df['model'] == model_name) & (df['dataset'] == dataset) & (df['sparsity'] == sparsity)]) > 0:
            continue
            
        new_row = {
            "model": model_name,
            "dataset": dataset,
            "recall@1": recalls[1],
            "recall@5": recalls[5], 
            "recall@10": recalls[10],
            "flops": flops,
            "sparsity": sparsity,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv("results/results.csv", index=False)

def add_results(config, module, results):
    if "baseline" in config["eval"]: 
        add_baseline_result(config, module, results)
    else: 
        add_posttrain_result(config, module, results)

def main(): 
    args = parse_args()
    config = load_config(args.config)
    model = _load_model(config, args)

    eval_module = EvaluateModule(model, **config["eval"]["evaluate_module"], sparsity=args.sparsity)
    trainer = pl.Trainer(**config["eval"]["trainer"])
    #trainer.test(eval_module)

    result = {'Pitts30k': {1: 92.38556338028168, 5: 96.30281690140845, 10: 97.37382629107981}, 'MSLS': {1: 88.19920606279321, 5: 93.75676651028509, 10: 95.10104655359076}, 'flops': 500, 'sparsity': args.sparsity}
    #results = eval_module.test_results 
    add_results(config, eval_module, result)



if __name__ == "__main__":
    main()