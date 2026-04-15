#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/ablations/measure_overhead/results}"
mkdir -p "$RESULTS_DIR"

python "$ROOT_DIR/code/experiments/efficientnetb0_stanfordcars_oui_timing_profile.py" --wd 5e-5 --seed 1 --output-dir "$RESULTS_DIR"
python "$ROOT_DIR/code/experiments/resnet50_food101_oui_timing_profile.py" --wd 5e-2 --seed 1 --output-dir "$RESULTS_DIR"
python "$ROOT_DIR/code/experiments/densenet121_cifar100_oui_timing_profile.py" --wd 5e-2 --seed 1 --output-dir "$RESULTS_DIR"
python "$ROOT_DIR/code/experiments/mobilenetv2_cifar10_oui_timing_profile.py" --wd 5e-4 --seed 1 --output-dir "$RESULTS_DIR"
