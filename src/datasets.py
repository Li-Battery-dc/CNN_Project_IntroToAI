from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .factory import build_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageListDataset(Dataset):
    def __init__(self, data_root: str | Path, entries: list[dict[str, Any]], transform=None) -> None:
        self.data_root = Path(data_root)
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        path = self.data_root / entry["path"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(entry["label"]), path.as_posix()


class RotationPredictionDataset(Dataset):
    angles = (0, 90, 180, 270)

    def __init__(self, data_root: str | Path, entries: list[dict[str, Any]], transform=None) -> None:
        self.data_root = Path(data_root)
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries) * len(self.angles)

    def __getitem__(self, index: int):
        entry = self.entries[index // len(self.angles)]
        rotation_label = index % len(self.angles)
        path = self.data_root / entry["path"]
        image = Image.open(path).convert("RGB")
        image = TF.rotate(image, self.angles[rotation_label])
        if self.transform is not None:
            image = self.transform(image)
        return image, rotation_label, path.as_posix()


class TwoViewImageDataset(Dataset):
    def __init__(self, data_root: str | Path, entries: list[dict[str, Any]], transform) -> None:
        self.data_root = Path(data_root)
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        path = self.data_root / entry["path"]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.transform(image), path.as_posix()


def load_split_file(split_file: str | Path) -> dict[str, Any]:
    with Path(split_file).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _limit_entries_per_class(entries: list[dict[str, Any]], limit_per_class: int | None) -> list[dict[str, Any]]:
    if limit_per_class is None:
        return entries
    seen: dict[int, int] = defaultdict(int)
    limited = []
    for entry in entries:
        label = int(entry["label"])
        if seen[label] < limit_per_class:
            limited.append(entry)
            seen[label] += 1
    return limited


def _scan_test_entries(data_root: Path, class_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    entries, _ = scan_classification_entries(data_root, "test", class_to_idx)
    return entries


def scan_classification_entries(
    data_root: str | Path,
    split: str,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    data_root = Path(data_root)
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing {split} directory: {split_root}")

    if class_to_idx is None:
        class_names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    else:
        class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    entries = []
    for class_name in class_names:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files:
            entries.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "label": class_to_idx[class_name],
                    "class": class_name,
                }
            )
    return entries, class_names


def class_names_from_split(split_data: dict[str, Any]) -> list[str]:
    class_to_idx = split_data["class_to_idx"]
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def build_dataset(
    data_root: str | Path,
    split_file: str | Path,
    split: str,
    transform_spec: dict[str, Any],
    image_size: int,
    limit_per_class: int | None = None,
) -> tuple[ImageListDataset, list[str]]:
    split_data = load_split_file(split_file)
    class_names = class_names_from_split(split_data)
    if split in {"train", "valid"}:
        entries = split_data["splits"][split]
    elif split == "test":
        entries = _scan_test_entries(Path(data_root), split_data["class_to_idx"])
    else:
        raise ValueError(f"Unknown split {split!r}; expected train, valid, or test")

    entries = _limit_entries_per_class(entries, limit_per_class)
    transform = build_transform(transform_spec, split=split, image_size=image_size)
    return ImageListDataset(data_root, entries, transform), class_names


def build_loader(
    data_root: str | Path,
    split_file: str | Path,
    split: str,
    transform_spec: dict[str, Any],
    image_size: int,
    batch_size: int,
    num_workers: int,
    limit_per_class: int | None = None,
    shuffle: bool | None = None,
) -> tuple[DataLoader, list[str]]:
    dataset, class_names = build_dataset(
        data_root=data_root,
        split_file=split_file,
        split=split,
        transform_spec=transform_spec,
        image_size=image_size,
        limit_per_class=limit_per_class,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, class_names


def build_full_label_dataset(
    data_root: str | Path,
    split: str,
    transform_spec: dict[str, Any],
    image_size: int,
    split_file: str | Path | None = None,
    limit_per_class: int | None = None,
) -> tuple[ImageListDataset, list[str]]:
    class_to_idx = None
    if split_file is not None:
        class_to_idx = load_split_file(split_file)["class_to_idx"]
    entries, class_names = scan_classification_entries(data_root, split, class_to_idx)
    entries = _limit_entries_per_class(entries, limit_per_class)
    transform = build_transform(transform_spec, split=split, image_size=image_size)
    return ImageListDataset(data_root, entries, transform), class_names


def build_full_label_loader(
    data_root: str | Path,
    split: str,
    transform_spec: dict[str, Any],
    image_size: int,
    batch_size: int,
    num_workers: int,
    split_file: str | Path | None = None,
    limit_per_class: int | None = None,
    shuffle: bool | None = None,
) -> tuple[DataLoader, list[str]]:
    dataset, class_names = build_full_label_dataset(
        data_root=data_root,
        split=split,
        transform_spec=transform_spec,
        image_size=image_size,
        split_file=split_file,
        limit_per_class=limit_per_class,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, class_names
