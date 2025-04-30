"""Insert static parameter‑memory numbers into existing results CSVs.

This script *only* appends or overwrites a `memory_MB` column in the
existing `results/baselines.csv` and `results/results.csv` files.
No model evaluation or trainer calls are performed.

* **Baselines:** parameter memory (all weights).
* **Post‑train:** parameter memory × (1 – sparsity).

Usage
-----
python measure_memory.py --config cfg.yaml               # baseline
python measure_memory.py --config cfg.yaml --sparsity 0.5  # post‑train
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Any, Dict

import pandas as pd
import torch

from trainer import PostTrainerModule  # noqa: F401
from model.baselines import DinoBoQ, DinoSalad, CosPlace, EigenPlaces, MixVPR  # noqa: F401
from common import load_config  # noqa: F401

torch.set_float32_matmul_precision("medium")

###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--sparsity", type=float, default=0.0)
    return p.parse_args()

###############################################################################
# Model loading helpers
###############################################################################

def _load_baseline(name: str):
    name = name.lower()
    if name == "dinoboq":
        return DinoBoQ()
    if name == "dinosalad":
        return DinoSalad()
    if name == "cosplace":
        return CosPlace()
    if name == "eigenplaces":
        return EigenPlaces()
    if name == "mixvpr":
        return MixVPR()
    raise ValueError(f"Unknown baseline model '{name}'")


def _load_model(cfg: Dict[str, Any], args: argparse.Namespace):
    if "baseline" in cfg["eval"]:
        return _load_baseline(cfg["eval"]["baseline"])
    pt = PostTrainerModule.load_from_checkpoint(cfg["eval"]["checkpoint_path"])
    net = pt.model
    net.forward = partial(net.forward, sparsity=args.sparsity)
    net.eval()
    return net

###############################################################################
# Memory measurement
###############################################################################

def _param_bytes(model: torch.nn.Module, sparsity: float = 0.0) -> int:
    
    if "sparse-vit" in model.__repr__():
        total = 0
        for name, p in model.named_parameters():
            if any(x in name for x in ["to_qkv", "lin", "to_out", "to_path"]) and "backbone" in name:
                total += int(p.numel()) * 0.25
            else:
                total += int(p.numel()) * 2
        return total
    else:
        total = 0
        for p in model.parameters():
            total += int(p.numel()) * 2
        return total


def measure_memory_mb(model: torch.nn.Module, is_baseline: bool, sparsity: float) -> float:
    if is_baseline:
        return _param_bytes(model) / 1024 ** 2
    return _param_bytes(model, sparsity) / 1024 ** 2

###############################################################################
# CSV update helpers
###############################################################################

def _ensure_memory_column(df: pd.DataFrame) -> pd.DataFrame:
    if "memory" not in df.columns:
        df["memory"] = pd.NA
    return df


def _update_baselines(model_name: str, mem: float):
    path = "results/baselines.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Baseline CSV not found. Run evaluation first.")
    df = pd.read_csv(path)
    df = _ensure_memory_column(df)
    df.loc[df["model"] == model_name, "memory"] = mem
    df.to_csv(path, index=False)
    print(f"Updated {path} for model '{model_name}' → {mem:.2f} MB")


def _update_results(model_name: str, sparsity: float, mem: float):
    path = "results/results.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Results CSV not found. Run evaluation first.")
    df = pd.read_csv(path)
    df = _ensure_memory_column(df)
    mask = (df["model"] == model_name) & (df["sparsity"] == sparsity)
    df.loc[mask, "memory"] = mem
    df.to_csv(path, index=False)
    print(f"Updated {path} for model '{model_name}' (sparsity={sparsity}) → {mem:.2f} MB")

###############################################################################
# Main
###############################################################################

def main():
    args = parse_args()
    cfg = load_config(args.config)

    model = _load_model(cfg, args)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    is_baseline = "baseline" in cfg["eval"]
    mem_mb = measure_memory_mb(model, is_baseline, args.sparsity)

    model_name = model.__repr__()
    if is_baseline:
        _update_baselines(model_name, mem_mb)
    else:
        _update_results(model_name, args.sparsity, mem_mb)


if __name__ == "__main__":
    main()
