from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.report_utils import load_json, markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="reports/assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(Path(args.runs).glob("*_*")):
        if not run_dir.is_dir():
            continue
        row = {"run": run_dir.name}
        best = run_dir / "best.json"
        test = run_dir / "test_metrics.json"
        if best.exists():
            row["best_valid_accuracy"] = load_json(best)["best_valid_accuracy"]
        if test.exists():
            metrics = load_json(test)
            row["test_accuracy"] = metrics["accuracy"]
            row["test_micro_f1"] = metrics["micro_f1"]
            row["test_macro_f1"] = metrics["macro_f1"]
            row["test_weighted_f1"] = metrics["weighted_f1"]
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "experiment_summary.csv", index=False)
    (out / "experiment_summary.md").write_text(markdown_table(frame), encoding="utf-8")
    print(f"Wrote {out / 'experiment_summary.csv'} and {out / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
