from __future__ import annotations

import argparse
from typing import Any

import torch
from tqdm import tqdm

from scripts.report_utils import save_classification_outputs, save_json, timestamp_run_dir, write_metric_history
from src.config import load_experiment_config
from src.datasets import build_full_label_loader
from src.engine import collect_predictions
from src.factory import build_loss, build_model
from src.self_supervised import apply_cosine_lr, init_lr_schedule, load_encoder_state, set_encoder_requires_grad, set_encoder_train_mode
from src.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/self_supervised.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pretrained")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-per-class", type=int)
    return parser.parse_args()


def _override_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in ("data_root", "epochs", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.pretrained is not None:
        config["pretrained"] = args.pretrained
    return config


def _load_pretrained_encoder(model, pretrained: str | None, device) -> None:
    if not pretrained:
        return
    checkpoint = torch.load(pretrained, map_location=device, weights_only=False)
    load_encoder_state(model, checkpoint["encoder_state"])


def _build_finetune_optimizer(model, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = dict(config["optimizer"])
    name = optimizer_config.pop("name")
    if name != "adamw":
        raise KeyError(f"finetune_ssl only supports adamw, got {name!r}")

    encoder_params = list(model.features.parameters()) + list(model.pool.parameters())
    classifier_params = list(model.classifier.parameters())
    param_groups = [
        {"params": encoder_params, "lr": float(config["encoder_lr"])},
        {"params": classifier_params, "lr": float(config["classifier_lr"])},
    ]
    return torch.optim.AdamW(param_groups, **optimizer_config)


def _train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp: bool, epoch: int, encoder_frozen: bool) -> dict[str, float]:
    model.train()
    if encoder_frozen:
        set_encoder_train_mode(model, train=False)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = tqdm(loader, desc=f"epoch {epoch:03d} finetune", leave=False)
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


def main() -> None:
    args = parse_args()
    config = _override_config(load_experiment_config(args.config, args.experiment), args)
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    run_dir = timestamp_run_dir(config["output_dir"], args.experiment)
    save_json(config, run_dir / "effective_config.json")

    train_loader, class_names = build_full_label_loader(
        config["data_root"],
        "train",
        config["transform"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        split_file=config.get("split_file"),
        limit_per_class=args.limit_per_class,
        shuffle=True,
    )
    test_loader, _ = build_full_label_loader(
        config["data_root"],
        "test",
        config["transform"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        split_file=config.get("split_file"),
        shuffle=False,
    )

    model = build_model(config["model"], num_classes=len(class_names)).to(device)
    pretrained = config.get("pretrained")
    if pretrained:
        _load_pretrained_encoder(model, pretrained, device)
    freeze_epochs = int(config.get("freeze_epochs", 0)) if pretrained else 0
    set_encoder_requires_grad(model, requires_grad=freeze_epochs == 0)

    loss_fn = build_loss(config["loss"])
    optimizer = _build_finetune_optimizer(model, config)
    init_lr_schedule(optimizer)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"pretrained={pretrained or 'none'}")

    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_encoder_requires_grad(model, requires_grad=True)
        encoder_frozen = freeze_epochs > 0 and epoch <= freeze_epochs
        lrs = apply_cosine_lr(
            optimizer,
            epoch=epoch,
            epochs=int(config["epochs"]),
            warmup_epochs=int(config.get("warmup_epochs", 0)),
        )
        metrics = _train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp, epoch, encoder_frozen)
        row = {
            "epoch": epoch,
            **metrics,
            "lr_encoder": lrs[0],
            "lr_classifier": lrs[1],
            "encoder_frozen": encoder_frozen,
        }
        history.append(row)
        write_metric_history(history, run_dir)
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

    checkpoint = {
        "experiment": args.experiment,
        "config": config,
        "class_names": class_names,
        "epoch": int(config["epochs"]),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, run_dir / "final.pt")
    torch.save(checkpoint, run_dir / "best.pt")

    result = collect_predictions(model, test_loader, loss_fn, device)
    report = save_classification_outputs(
        result["labels"],
        result["preds"],
        class_names,
        run_dir,
        "test",
        {"loss": result["loss"], "accuracy": result["accuracy"]},
    )
    save_json(
        {
            "split": "test",
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "micro_f1": report["micro avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "checkpoint": (run_dir / "final.pt").as_posix(),
        },
        run_dir / "test_metrics.json",
    )
    print(f"{run_dir.name} test: acc={result['accuracy']:.4f}, macro_f1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
