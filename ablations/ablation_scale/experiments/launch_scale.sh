#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER="${LAUNCHER:-$ROOT_DIR/main_experiments/experiments/launch_main_generic.sh}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/ablations/ablation_scale/results}"
mkdir -p "$RESULTS_DIR"

SCALES=(
  "0.6666 5.0"
  "0.6666 3.0"
  "0.3333 3.0"
  "0.3333 5.0"
  "0.5 2.0"
)

for pair in "${SCALES[@]}"; do
  read -r S1 S2 <<< "$pair"
  CASE=resnet50_food101   METHOD=ouidecay   OPTIMIZER=adamw   WD=5e-2   SEED=1   WD_MIN_RATIO="$S1"   WD_MAX_RATIO="$S2"   RUN_TAG="ouidecay_scale_s1_${S1}_s2_${S2}"   OUTPUT_DIR="$RESULTS_DIR"   bash "$LAUNCHER"
done
