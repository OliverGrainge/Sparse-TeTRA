import torch


def EigenPlaces():
    return torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )
