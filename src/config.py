from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(config_path: str | Path, experiment: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    defaults = raw.get("defaults", {})
    experiments = raw.get("experiments", {})
    if experiment not in experiments:
        available = ", ".join(sorted(experiments))
        raise KeyError(f"Unknown experiment {experiment!r}. Available: {available}")

    config = deep_merge(defaults, experiments[experiment] or {})
    config["experiment"] = experiment
    return config
