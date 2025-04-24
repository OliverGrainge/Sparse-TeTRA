import torch
import contextlib
import io
import logging


def ResNet50BoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq", 
            backbone_name="resnet50",
            output_dim=16384,
        )
    return model


def DinoBoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2", 
            output_dim=12288,
        )
    original_forward = model.forward  # ← Save before overwriting
    model.forward = lambda x: original_forward(x)[0]  # safe wrapper
    return model


def EigenPlaces():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )
    return model
