from __future__ import annotations

import math
from typing import Any


class CosineLRScheduler:
    def __init__(self, optimizer, epochs: int, min_lr: float = 1.0e-5) -> None:
        self.optimizer = optimizer
        self.epochs = max(int(epochs), 1)
        self.min_lr = float(min_lr)
        self.initial_lr = float(optimizer.lr)

    def step(self, completed_epoch: int) -> None:
        progress = min(max(completed_epoch / self.epochs, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.optimizer.lr = self.min_lr + (self.initial_lr - self.min_lr) * scale


def build_scheduler(spec: dict[str, Any] | None, optimizer, epochs: int):
    if not spec:
        return None
    spec = dict(spec)
    name = spec.pop("name")
    if name == "cosine":
        return CosineLRScheduler(optimizer, epochs=epochs, **spec)
    raise KeyError(f"Unknown scheduler {name!r}")
