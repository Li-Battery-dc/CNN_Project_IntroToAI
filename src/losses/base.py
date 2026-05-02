from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
