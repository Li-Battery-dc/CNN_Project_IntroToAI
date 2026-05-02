from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


ENCODER_PREFIXES = ("features.", "pool.")


class RotationPredictionModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feature_dim, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(images))


class SimCLRModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.projector(features)


def encoder_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in backbone.state_dict().items()
        if key.startswith(ENCODER_PREFIXES)
    }


def load_encoder_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key.startswith(ENCODER_PREFIXES)]
    missing_encoder = [key for key in missing if key.startswith(ENCODER_PREFIXES)]
    if unexpected or missing_encoder:
        raise RuntimeError(f"Could not load encoder state. missing={missing_encoder}, unexpected={unexpected}")


def set_encoder_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for module_name in ("features", "pool"):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def set_encoder_train_mode(model: nn.Module, train: bool) -> None:
    model.features.train(train)
    model.pool.train(train)


def nt_xent_loss(projections: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, float]]:
    projections = F.normalize(projections.float(), dim=1)
    batch_size = projections.size(0) // 2
    logits = projections @ projections.T / temperature
    logits = logits.masked_fill(torch.eye(logits.size(0), device=logits.device, dtype=torch.bool), -1.0e9)
    targets = torch.arange(logits.size(0), device=logits.device)
    targets = (targets + batch_size) % logits.size(0)
    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        positive_cosine = F.cosine_similarity(projections[:batch_size], projections[batch_size:]).mean().item()
        metrics = {
            "positive_top1_accuracy": (preds == targets).float().mean().item(),
            "positive_cosine": positive_cosine,
        }
    return loss, metrics


class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
        eps: float = 1.0e-8,
        exclude_bias_norm: bool = True,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "eta": eta,
            "eps": eps,
            "exclude_bias_norm": exclude_bias_norm,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eta = group["eta"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                apply_lars = not (group["exclude_bias_norm"] and parameter.ndim == 1)
                if weight_decay and apply_lars:
                    update = update.add(parameter, alpha=weight_decay)

                if apply_lars:
                    weight_norm = torch.linalg.vector_norm(parameter)
                    update_norm = torch.linalg.vector_norm(update)
                    if weight_norm > 0 and update_norm > 0:
                        update = update * (eta * weight_norm / (update_norm + eps))

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(update, alpha=group["lr"])
                parameter.add_(buffer, alpha=-1.0)
        return loss


def build_torch_optimizer(spec: dict[str, Any], params) -> torch.optim.Optimizer:
    spec = dict(spec)
    name = spec.pop("name")
    if name == "sgd":
        return torch.optim.SGD(params, **spec)
    if name == "adamw":
        return torch.optim.AdamW(params, **spec)
    if name == "lars":
        return LARS(params, **spec)
    raise KeyError(f"Unknown torch optimizer {name!r}")


def init_lr_schedule(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])


def apply_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    warmup_epochs: int = 0,
) -> list[float]:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        multiplier = epoch / warmup_epochs
    else:
        denominator = max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denominator))
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    lrs = []
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
        lrs.append(group["lr"])
    return lrs
