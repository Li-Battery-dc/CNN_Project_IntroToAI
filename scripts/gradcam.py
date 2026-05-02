from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.report_utils import load_json
from src.datasets import build_dataset
from src.factory import build_model
from src.transforms import denormalize
from src.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--num-correct", type=int, default=4)
    parser.add_argument("--num-wrong", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def pick_examples(model, dataset, device, num_correct: int, num_wrong: int) -> list[tuple[int, int]]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    picked, correct, wrong, offset = [], 0, 0, 0
    for images, targets, _ in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        for i, (pred, target) in enumerate(zip(preds.tolist(), targets.tolist())):
            if pred == target and correct < num_correct:
                picked.append((offset + i, pred))
                correct += 1
            elif pred != target and wrong < num_wrong:
                picked.append((offset + i, pred))
                wrong += 1
        if correct >= num_correct and wrong >= num_wrong:
            break
        offset += len(targets)
    return picked


def gradcam(model, image: torch.Tensor, target_class: int, device) -> np.ndarray:
    layer = model.get_cam_target_layer()
    saved = {}
    fwd = layer.register_forward_hook(lambda _m, _i, out: saved.update(act=out.detach()))
    bwd = layer.register_full_backward_hook(lambda _m, _gi, go: saved.update(grad=go[0].detach()))
    model.zero_grad(set_to_none=True)
    score = model(image[None].to(device))[0, target_class]
    score.backward()
    fwd.remove()
    bwd.remove()

    weights = saved["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * saved["act"]).sum(dim=1))
    cam = torch.nn.functional.interpolate(cam[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1.0e-8)
    return cam.cpu().numpy()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "effective_config.json")
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])

    dataset, class_names = build_dataset(
        config["data_root"],
        config["split_file"],
        args.split,
        config["transform"],
        int(config["image_size"]),
    )
    model = build_model(config["model"], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    examples = pick_examples(model, dataset, device, args.num_correct, args.num_wrong)
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 3 * len(examples)), dpi=150)
    axes = np.asarray([axes]) if len(examples) == 1 else axes
    for row, (index, pred) in enumerate(examples):
        image, target, path = dataset[index]
        display = denormalize(image).permute(1, 2, 0).cpu().numpy()
        heatmap = gradcam(model, image, pred, device)
        axes[row, 0].imshow(display)
        axes[row, 0].set_title(f"true={class_names[target]}\npred={class_names[pred]}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(display)
        axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[row, 1].set_title(Path(path).name)
        axes[row, 1].axis("off")
    fig.tight_layout()
    out = run_dir / f"{args.split}_gradcam.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
