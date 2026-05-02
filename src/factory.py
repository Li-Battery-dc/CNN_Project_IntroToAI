from __future__ import annotations

from typing import Any

from .losses import CrossEntropyLoss, FocalLoss, LabelSmoothingCrossEntropy
from .models import BasicCNN, RegularizedCNN
from .optimizers import Adam, SGD
from .transforms import basic_transform, rotation_pretrain_transform, simclr_pretrain_transform, train_aug_transform


def _kwargs(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    spec = dict(spec)
    return spec.pop("name"), spec


def build_model(spec: dict[str, Any], num_classes: int):
    name, kwargs = _kwargs(spec)
    cls = {"basic_cnn": BasicCNN, "regularized_cnn": RegularizedCNN}[name]
    return cls(num_classes=num_classes, **kwargs)


def build_optimizer(spec: dict[str, Any], params):
    name, kwargs = _kwargs(spec)
    cls = {"sgd": SGD, "adam": Adam}[name]
    return cls(params=params, **kwargs)


def build_loss(spec: dict[str, Any]):
    name, kwargs = _kwargs(spec)
    cls = {
        "cross_entropy": CrossEntropyLoss,
        "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
        "focal_loss": FocalLoss,
    }[name]
    return cls(**kwargs)


def build_transform(spec: dict[str, Any], split: str, image_size: int):
    name, _ = _kwargs(spec)
    fn = {
        "basic": basic_transform,
        "train_aug": train_aug_transform,
        "rotation_pretrain": rotation_pretrain_transform,
        "simclr_pretrain": simclr_pretrain_transform,
    }[name]
    return fn(split=split, image_size=image_size)
