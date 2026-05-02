from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 0:
                loss = loss * alpha
            else:
                loss = loss * alpha.gather(0, targets)

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
