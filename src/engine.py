from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from tqdm import tqdm


def unpack_batch(batch: Iterable[Any]):
    images, targets = batch[0], batch[1]
    paths = batch[2] if len(batch) > 2 else None
    return images, targets, paths


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, loss_fn, optimizer, device: torch.device, epoch: int) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)
    for batch in progress:
        images, targets, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size
        progress.set_postfix(loss=total_loss / total_count, acc=total_correct / total_count)

    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device: torch.device, desc: str = "eval") -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        images, targets, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size

    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


@torch.no_grad()
def collect_predictions(model, loader, loss_fn, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    labels: list[int] = []
    preds: list[int] = []
    paths: list[str] = []

    for batch in tqdm(loader, desc="predict", leave=False):
        images, targets, batch_paths = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, targets)
        batch_preds = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (batch_preds == targets).sum().item()
        total_count += batch_size

        labels.extend(targets.cpu().tolist())
        preds.extend(batch_preds.cpu().tolist())
        if batch_paths is not None:
            paths.extend(list(batch_paths))

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "labels": labels,
        "preds": preds,
        "paths": paths,
    }
