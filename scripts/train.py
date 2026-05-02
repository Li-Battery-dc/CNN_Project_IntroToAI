from __future__ import annotations

import argparse

import torch

from scripts.report_utils import save_json, timestamp_run_dir, write_history
from src.config import load_experiment_config
from src.datasets import build_loader
from src.engine import evaluate_epoch, train_one_epoch
from src.factory import build_loss, build_model, build_optimizer
from src.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split-file")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.experiment)
    for key in ("data_root", "split_file", "epochs", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    run_dir = timestamp_run_dir(config["output_dir"], args.experiment)
    checkpoint_path = run_dir / "best.pt"
    save_json(config, run_dir / "effective_config.json")

    train_loader, class_names = build_loader(
        config["data_root"],
        config["split_file"],
        "train",
        config["transform"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        limit_per_class=args.limit_per_class,
    )
    valid_loader, _ = build_loader(
        config["data_root"],
        config["split_file"],
        "valid",
        config["transform"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        limit_per_class=args.limit_per_class,
        shuffle=False,
    )

    model = build_model(config["model"], num_classes=len(class_names)).to(device)
    loss_fn = build_loss(config["loss"])
    optimizer = build_optimizer(config["optimizer"], params=model.parameters())

    print(f"run_dir={run_dir}")
    print(f"device={device}")
    history, best_acc = [], -1.0
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        valid_metrics = evaluate_epoch(model, valid_loader, loss_fn, device, desc=f"epoch {epoch:03d} valid")
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(row)
        write_history(history, run_dir)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_acc={row['valid_accuracy']:.4f}"
        )
        if row["valid_accuracy"] > best_acc:
            best_acc = row["valid_accuracy"]
            torch.save(
                {
                    "experiment": args.experiment,
                    "config": config,
                    "class_names": class_names,
                    "epoch": epoch,
                    "best_valid_accuracy": best_acc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    save_json({"best_valid_accuracy": best_acc, "checkpoint": checkpoint_path.as_posix()}, run_dir / "best.json")
    print(f"best_valid_accuracy={best_acc:.4f}")


if __name__ == "__main__":
    main()
