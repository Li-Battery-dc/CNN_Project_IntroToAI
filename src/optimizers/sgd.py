from __future__ import annotations

from typing import Iterable

import torch

from .base import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float = 0.0) -> None:
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay:
                grad = grad.add(param, alpha=self.weight_decay)
            param.add_(grad, alpha=-self.lr)

    def state_dict(self) -> dict:
        return {"lr": self.lr, "weight_decay": self.weight_decay}
