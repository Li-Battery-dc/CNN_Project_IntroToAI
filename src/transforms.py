from __future__ import annotations

import torch
from torchvision import transforms as T

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


def _eval_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


def basic_transform(split: str, image_size: int = 96) -> T.Compose:
    return _eval_transform(image_size)


def train_aug_transform(split: str, image_size: int = 96) -> T.Compose:
    if split != "train":
        return _eval_transform(image_size)
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(STL10_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(STL10_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)
