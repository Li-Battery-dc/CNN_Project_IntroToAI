from __future__ import annotations

import torch
from torch import nn

class BasicCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.feature_dim = 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_cam_target_layer(self) -> nn.Module:
        return _last_conv_layer(self.features)


class RegularizedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.feature_dim = 128
        self.features = nn.Sequential(
            _conv_bn_relu(3, 32),
            _conv_bn_relu(32, 32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.5),
            _conv_bn_relu(32, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.75),
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_cam_target_layer(self) -> nn.Module:
        return _last_conv_layer(self.features)


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


def _last_conv_layer(module: nn.Module) -> nn.Module:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return child
    raise RuntimeError("No Conv2d layer found for Grad-CAM")
