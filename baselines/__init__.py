from .boq import DinoBoQ, ResNet50BoQ
from .eigenplaces import EigenPlaces

__all__ = ["EigenPlaces", "ResNet50BoQ", "DinoBoQ"]

ALL_BASELINES = {
    "eigenplaces": EigenPlaces,
    "resnet50_boq": ResNet50BoQ,
    "dino_boq": DinoBoQ,
}

IMAGE_SIZES = {
    "eigenplaces": (512, 512),
    "resnet50_boq": (384, 384),
    "dino_boq": (322, 322),
}
