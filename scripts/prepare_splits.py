from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="STL10")
    parser.add_argument("--out", default="splits/stl10_seed42_valid20.json")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rng = random.Random(args.seed)
    classes = sorted(path.name for path in (data_root / "train").iterdir() if path.is_dir())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    splits = {"train": [], "valid": []}
    counts = {}

    for name in classes:
        files = sorted(path for path in (data_root / "train" / name).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(files)
        n_valid = round(len(files) * args.valid_ratio)
        split_files = {"valid": sorted(files[:n_valid]), "train": sorted(files[n_valid:])}
        counts[name] = {split: len(paths) for split, paths in split_files.items()}
        for split, paths in split_files.items():
            for path in paths:
                splits[split].append(
                    {
                        "path": path.relative_to(data_root).as_posix(),
                        "label": class_to_idx[name],
                        "class": name,
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "valid_ratio": args.valid_ratio,
                "class_to_idx": class_to_idx,
                "counts": counts,
                "splits": splits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}: train={len(splits['train'])}, valid={len(splits['valid'])}")


if __name__ == "__main__":
    main()
