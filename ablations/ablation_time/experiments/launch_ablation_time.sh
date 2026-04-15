#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER="${LAUNCHER:-$ROOT_DIR/main_experiments/experiments/launch_main_generic.sh}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/ablations/ablation_time/results}"
GAPS=(1 4 16 64 128 256 512 1024)

mkdir -p "$RESULTS_DIR"

for gap in "${GAPS[@]}"; do
  CASE=efficientnetb0_stanfordcars   METHOD=ouidecay   OPTIMIZER=adam   WD=5e-5   SEED=1   UNBALANCED_WD_EVERY="$gap"   RUN_TAG="efficientnetb0_gap_pow2_gap${gap}"   OUTPUT_DIR="$RESULTS_DIR"   bash "$LAUNCHER"
done
