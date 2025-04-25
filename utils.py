import argparse
import importlib
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> Dict[Any, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_lightning2model_checkpoint(model: nn.Module, checkpoint_path: str):
    sd = torch.load(checkpoint_path)
    new_sd = {}
    for key in sd["state_dict"].keys():
        new_sd[key.replace("model.", "")] = sd["state_dict"][key]
    model.load_state_dict(new_sd)
    return model


def import_model_cls(baseline_name):
    candidate_modules = ["model.baselines", "model.models"]
    for module_name in candidate_modules:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            if attr.lower() == baseline_name.lower():
                return getattr(module, attr)

    raise ImportError(
        f"Function '{baseline_name}' not found in any of {candidate_modules} (case-insensitive)"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    return parser.parse_args()


def pair(t):
    return (
        (t, t)
        if isinstance(t, int)
        else tuple(t[:2]) if isinstance(t, (list, tuple)) else (t, t)
    )
