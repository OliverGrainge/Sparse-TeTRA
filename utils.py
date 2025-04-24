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
