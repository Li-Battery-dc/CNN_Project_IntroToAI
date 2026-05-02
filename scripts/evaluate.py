from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.report_utils import load_json, save_classification_outputs, save_json
from src.datasets import build_loader
from src.engine import collect_predictions
from src.factory import build_loss, build_model
from src.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--limit-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "effective_config.json")
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])

    loader, class_names = build_loader(
        config["data_root"],
        config["split_file"],
        args.split,
        config["transform"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        limit_per_class=args.limit_per_class,
        shuffle=False,
    )
    model = build_model(config["model"], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = collect_predictions(model, loader, build_loss(config["loss"]), device)
    report = save_classification_outputs(
        result["labels"],
        result["preds"],
        class_names,
        run_dir,
        args.split,
        {"loss": result["loss"], "accuracy": result["accuracy"]},
    )
    save_json(
        {
            "split": args.split,
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "micro_f1": report["micro avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        },
        run_dir / f"{args.split}_metrics.json",
    )
    print(f"{run_dir.name} {args.split}: acc={result['accuracy']:.4f}, macro_f1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
