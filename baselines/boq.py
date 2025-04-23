import torch


def ResNet50BoQ():
    return torch.hub.load(
        "amaralibey/bag-of-queries",
        "get_trained_boq",
        backbone_name="resnet50",
        output_dim=16384,
    )


def DinoBoQ():
    return torch.hub.load(
        "amaralibey/bag-of-queries",
        "get_trained_boq",
        backbone_name="dinov2",
        output_dim=12288,
    )
