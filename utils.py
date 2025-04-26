from typing import Any, Dict
import torch.nn as nn 
import os 
import torch 
import yaml
from model import * 


def load_config(config_path: str) -> Dict[Any, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_pretrain_checkpoint2model(model: nn.Module, checkpoint_path: str):
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def load_posttrain_checkpoint2model(model: nn.Module, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model


def _load_model_module(model_name: str): 
    if model_name.lower() == 'vit': 
        return ViT
    else: 
        raise ValueError(f"Model {model_name} not found")

def load_model(model_name: str, model_init_args: dict): 
    model_module = _load_model_module(model_name)
    model = model_module(**model_init_args)
    return model
    