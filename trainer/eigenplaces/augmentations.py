from typing import Tuple, Union

import torch
import torchvision.transforms as T


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert (
            len(images.shape) == 4
        ), f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale, antialias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert (
            len(images.shape) == 4
        ), f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images
