#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiments.yaml}
EXPERIMENTS=(
  baseline_sgd_ce
  optimizer_adam
  loss_label_smooth
  data_aug
  bn_dropout
  baseline_strong
  vgg_medium_tuned
  combined_best
)

for EXP in "${EXPERIMENTS[@]}"; do
  python -m scripts.train --config "$CONFIG" --experiment "$EXP"
done

BEST_RUN=$(ls -td runs/*_combined_best | head -n 1)
python -m scripts.evaluate --run-dir "$BEST_RUN" --split test
python -m scripts.gradcam --run-dir "$BEST_RUN" --split test
python -m scripts.summarize_results --runs runs --out reports/assets
