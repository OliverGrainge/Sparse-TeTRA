from typing import Any, Dict
import torch.nn as nn 
import os 
import torch 
import yaml
from model import * 
from model.aggregation import * 


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
    new_sd = {}
    for key, value in state_dict.items(): 
        if "teacher" not in key and "projector" not in key: 
            new_sd[key.replace("model.", "")] = value
    model.load_state_dict(new_sd, strict=True)
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
    

def load_agg_method(agg_method: str): 
    if agg_method.lower() == 'cls': 
        return CLS
    else: 
        raise ValueError(f"Aggregation method {agg_method} not found")
