from __future__ import annotations

from typing import Iterable

import torch

from .base import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay:
                grad = grad.add(param, alpha=self.weight_decay)
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            param.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "betas": [self.beta1, self.beta2],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
        }
