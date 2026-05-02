from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch


class BaseOptimizer(ABC):
    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        self.params = [param for param in params if param.requires_grad]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}
