from trainer import EvaluateModule, PostTrainerModule
import pytorch_lightning as pl
import argparse
import yaml 
from typing import Any, Dict
from model.baselines import DinoBoQ, DinoSalad, CosPlace, EigenPlaces, MixVPR
from functools import partial
import os
import pandas as pd
import torch 
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
    elif model_name.lower() == "cosplace": 
        return CosPlace()
    elif model_name.lower() == "eigenplaces": 
        return EigenPlaces()
    elif model_name.lower() == "mixvpr": 
        return MixVPR()
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
        mask = (df["model"] == model_name) & (df["dataset"] == dataset)
        if not df.loc[mask].empty:
            print(f'========> SKIPPING')
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

def add_posttrain_result(config, args, module, result): 
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
    sparsity = args.sparsity
    for dataset, recalls in result.items():
        # Check if combination already exists
        mask = (df["model"] == model_name) & (df["dataset"] == dataset) & (df['sparsity'] == sparsity)
        if not df.loc[mask].empty:
            print(f'========> SKIPPING')
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

def add_results(config, args, module, results):
    if "baseline" in config["eval"]: 
        add_baseline_result(config, module, results)
    else: 
        add_posttrain_result(config, args,module, results)


def _compute_flops(model, config, args): 
    image_size = config["eval"]["evaluate_module"]["image_size"]
    input = torch.randn(1, 3, image_size, image_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(input)  # warm-up

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True) as prof:
        with torch.no_grad():
            model(input)

    events = prof.key_averages()
    flops = 0
    for evt in events:
        if "linear" in evt.key.lower() or "addmm" in evt.key.lower():   
            flops += evt.flops * (1 - args.sparsity)
        else:
            flops += evt.flops
    return flops

def main(): 
    args = parse_args()
    config = load_config(args.config)
    model = _load_model(config, args)
    flops = _compute_flops(model, config, args)
    eval_module = EvaluateModule(model, **config["eval"]["evaluate_module"], sparsity=args.sparsity)
    trainer = pl.Trainer(**config["eval"]["trainer"])
    trainer.test(eval_module)
    result = eval_module.test_results 
    
    result["flops"] = flops
    add_results(config, args, eval_module, result)



if __name__ == "__main__":
    main()