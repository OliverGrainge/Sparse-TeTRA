import contextlib
import io

import torch
import torch.nn as nn


def freeze(model): 
    for param in model.parameters(): 
        param.requires_grad = False
    return model


def DinoBoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
        )
        original_forward = model.forward  # ← Save before overwriting
        model.forward = lambda x: original_forward(x)[0]  # safe wrapper
        model = freeze(model)
        model = model.eval()
        model.__repr__ = lambda: "DINOv2-SALAD"
    return model


def DinoSalad(): 
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
       io.StringIO()
    ):
        model = torch.hub.load("serizba/salad", "dinov2_salad")
        model = nn.Sequential(model.backbone, model.aggregator)
        model = freeze(model)
        model = model.eval()
        model.__repr__ = lambda: "DINOv2-BoQ"
    return model


def CosPlace(): 
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
       io.StringIO()
    ):
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        model = freeze(model)
        model = model.eval()
        model.__repr__ = lambda: "CosPlace"
    return model


def EigenPlaces(): 
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
       io.StringIO()
    ):
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        model = freeze(model)
        model = model.eval()
        model.__repr__ = lambda: "EigenPlaces"
    return model


def MixVPR(): 
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
       io.StringIO()
    ):
        model = torch.hub.load("jarvisyjw/MixVPR", 'get_trained_model', pretrained=True)
        model = freeze(model)
        model = model.eval()
        model.__repr__ = lambda: "MixVPR"
    return model


