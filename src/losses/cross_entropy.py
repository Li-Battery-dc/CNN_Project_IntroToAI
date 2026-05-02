from __future__ import annotations

import torch

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -log_probs.gather(1, targets[:, None]).mean()


class LabelSmoothingCrossEntropy(BaseLoss):
    def __init__(self, smoothing: float = 0.1) -> None:
        self.smoothing = smoothing

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
        smooth = -log_probs.mean(dim=1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()
