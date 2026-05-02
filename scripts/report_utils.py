from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def timestamp_run_dir(root: str | Path, experiment: str) -> Path:
    run_dir = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{experiment}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_history(history: list[dict], run_dir: Path) -> None:
    save_json(history, run_dir / "history.json")
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, run_dir / "history.png")


def plot_history(history: list[dict], out_path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="valid")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["valid_accuracy"] for row in history], label="valid")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_classification_outputs(labels: list[int], preds: list[int], class_names: list[str], run_dir: Path, split: str, metrics: dict) -> dict:
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    report["micro avg"] = {"precision": p, "recall": r, "f1-score": f1, "support": len(labels)}
    report["summary"] = metrics

    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    save_json(report, run_dir / f"{split}_classification_report.json")
    pd.DataFrame(report).transpose().to_csv(run_dir / f"{split}_classification_report.csv")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(run_dir / f"{split}_confusion_matrix.csv")
    plot_confusion_matrix(matrix, class_names, run_dir / f"{split}_confusion_matrix.png")
    return report


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set(xlabel="Predicted", ylabel="True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No runs found.\n"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[col]) else str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"
