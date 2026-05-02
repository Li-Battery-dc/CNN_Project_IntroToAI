from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.report_utils import save_json, timestamp_run_dir, write_metric_history
from src.config import load_experiment_config
from src.datasets import RotationPredictionDataset, TwoViewImageDataset, scan_classification_entries
from src.factory import build_model, build_transform
from src.self_supervised import (
    RotationPredictionModel,
    SimCLRModel,
    apply_cosine_lr,
    build_torch_optimizer,
    encoder_state_dict,
    init_lr_schedule,
    nt_xent_loss,
)
from src.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/self_supervised.yaml")
    parser.add_argument("--method", required=True, choices=["rotation", "simclr"])
    parser.add_argument("--experiment")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-images", type=int)
    return parser.parse_args()


def _override_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in ("data_root", "epochs", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def _save_encoder_checkpoint(
    path: Path,
    method: str,
    config: dict[str, Any],
    epoch: int,
    monitor_value: float,
    model,
    optimizer,
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "method": method,
            "config": config,
            "epoch": epoch,
            "monitor_value": monitor_value,
            "encoder_state": encoder_state_dict(model.backbone),
            "ssl_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def _smooth_loss(history: list[dict[str, Any]], window: int) -> float:
    values = [row["train_loss"] for row in history[-window:]]
    return sum(values) / len(values)


def _train_rotation_epoch(model, loader, optimizer, scaler, device, use_amp: bool, epoch: int) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    progress = tqdm(loader, desc=f"epoch {epoch:03d} rotation", leave=False)
    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size
        progress.set_postfix(loss=total_loss / total_count, acc=total_correct / total_count)

    return {"train_loss": total_loss / total_count, "train_accuracy": total_correct / total_count}


def _train_simclr_epoch(model, loader, optimizer, scaler, device, use_amp: bool, epoch: int, temperature: float) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_cosine = 0.0
    total_count = 0

    progress = tqdm(loader, desc=f"epoch {epoch:03d} simclr", leave=False)
    for view1, view2, _ in progress:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        images = torch.cat([view1, view2], dim=0)
        batch_size = view1.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            projections = model(images)
        loss, metrics = nt_xent_loss(projections, temperature)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_acc += metrics["positive_top1_accuracy"] * batch_size
        total_cosine += metrics["positive_cosine"] * batch_size
        total_count += batch_size
        progress.set_postfix(loss=total_loss / total_count, top1=total_acc / total_count)

    return {
        "train_loss": total_loss / total_count,
        "positive_top1_accuracy": total_acc / total_count,
        "positive_cosine": total_cosine / total_count,
    }


def main() -> None:
    args = parse_args()
    experiment = args.experiment or f"{args.method}_pretrain"
    config = _override_config(load_experiment_config(args.config, experiment), args)
    if config["method"] != args.method:
        raise ValueError(f"Config experiment {experiment!r} uses method {config['method']!r}, not {args.method!r}")

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    run_dir = timestamp_run_dir(config["output_dir"], experiment)
    save_json(config, run_dir / "effective_config.json")

    class_to_idx = None
    if config.get("split_file"):
        from src.datasets import load_split_file

        class_to_idx = load_split_file(config["split_file"])["class_to_idx"]
    entries, class_names = scan_classification_entries(config["data_root"], "train", class_to_idx)
    if args.limit_images is not None:
        entries = entries[: args.limit_images]
    transform = build_transform(config["transform"], split="train", image_size=int(config["image_size"]))

    if args.method == "rotation":
        dataset = RotationPredictionDataset(config["data_root"], entries, transform)
        model = RotationPredictionModel(build_model(config["model"], num_classes=len(class_names))).to(device)
        drop_last = False
    else:
        dataset = TwoViewImageDataset(config["data_root"], entries, transform)
        projection = config["projection_head"]
        model = SimCLRModel(
            build_model(config["model"], num_classes=len(class_names)),
            hidden_dim=int(projection["hidden_dim"]),
            projection_dim=int(projection["projection_dim"]),
        ).to(device)
        drop_last = True

    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=drop_last,
    )
    optimizer = build_torch_optimizer(config["optimizer"], model.parameters())
    init_lr_schedule(optimizer)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"images={len(entries)} dataset_items={len(dataset)}")

    history, best_loss = [], float("inf")
    window = int(config.get("checkpoint", {}).get("moving_average_window", 5))
    for epoch in range(1, int(config["epochs"]) + 1):
        lrs = apply_cosine_lr(
            optimizer,
            epoch=epoch,
            epochs=int(config["epochs"]),
            warmup_epochs=int(config.get("warmup_epochs", 0)),
        )
        if args.method == "rotation":
            metrics = _train_rotation_epoch(model, loader, optimizer, scaler, device, use_amp, epoch)
        else:
            metrics = _train_simclr_epoch(
                model,
                loader,
                optimizer,
                scaler,
                device,
                use_amp,
                epoch,
                temperature=float(config["temperature"]),
            )

        row = {"epoch": epoch, **metrics, "lr": lrs[0]}
        history.append(row)
        write_metric_history(history, run_dir)
        smooth_loss = _smooth_loss(history, window)
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

        _save_encoder_checkpoint(run_dir / "last_encoder.pt", args.method, config, epoch, smooth_loss, model, optimizer, history)
        if smooth_loss < best_loss:
            best_loss = smooth_loss
            _save_encoder_checkpoint(run_dir / "encoder.pt", args.method, config, epoch, smooth_loss, model, optimizer, history)

    save_json({"best_smoothed_train_loss": best_loss, "checkpoint": (run_dir / "encoder.pt").as_posix()}, run_dir / "best.json")


if __name__ == "__main__":
    main()
