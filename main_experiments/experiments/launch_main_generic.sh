#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE="${CASE:?Set CASE to one of: efficientnetb0_stanfordcars, resnet50_food101, densenet121_cifar100, mobilenetv2_cifar10}"
METHOD="${METHOD:?Set METHOD to uniform, adadecay, adadecayg, or ouidecay}"
OPTIMIZER="${OPTIMIZER:-}"
WD="${WD:?Set WD to the desired base weight decay}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
MIN_LR="${MIN_LR:-}"
GRAD_CLIP="${GRAD_CLIP:-}"
RUN_TAG="${RUN_TAG:-public-main}"
DATASETS_ROOT="${DATASETS_ROOT:-$ROOT_DIR/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/main_experiments/results}"
ALPHADECAY_ROOT="${ALPHADECAY_ROOT:-$ROOT_DIR/code/vendor/alphadecay_llama}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ADAMW_FUSED_MODE="${ADAMW_FUSED_MODE:-auto}"
UNBALANCED_WD_EVERY="${UNBALANCED_WD_EVERY:-256}"
WD_MIN_RATIO="${WD_MIN_RATIO:-0.6666}"
WD_MAX_RATIO="${WD_MAX_RATIO:-5.0}"
OUI_WINDOW="${OUI_WINDOW:-5}"
OUI_SAMPLE_MODE="${OUI_SAMPLE_MODE:-random}"
ADACADECY_ALPHA="${ADACADECY_ALPHA:-4.0}"
ADACADECY_EPS="${ADACADECY_EPS:-1e-8}"

case "$CASE" in
  efficientnetb0_stanfordcars)
    EXP_SCRIPT="code/experiments/efficientnetb0_stanfordcars.py"
    OPTIMIZER="${OPTIMIZER:-adam}"
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    LR="${LR:-8e-4}"
    MIN_LR="${MIN_LR:-1e-5}"
    GRAD_CLIP="${GRAD_CLIP:-1.0}"
    DATA_ROOT="${DATA_ROOT:-$DATASETS_ROOT/stanford-cars}"
    ;;
  resnet50_food101)
    EXP_SCRIPT="code/experiments/resnet50_food101.py"
    OPTIMIZER="${OPTIMIZER:-adamw}"
    EPOCHS="${EPOCHS:-50}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    LR="${LR:-3e-4}"
    MIN_LR="${MIN_LR:-3e-5}"
    GRAD_CLIP="${GRAD_CLIP:-0}"
    DATA_ROOT="${DATA_ROOT:-$DATASETS_ROOT/food-101}"
    ;;
  densenet121_cifar100)
    EXP_SCRIPT="code/experiments/densenet121_cifar100.py"
    OPTIMIZER="${OPTIMIZER:-adamw}"
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    LR="${LR:-5e-4}"
    MIN_LR="${MIN_LR:-5e-6}"
    GRAD_CLIP="${GRAD_CLIP:-0}"
    DATA_ROOT="${DATA_ROOT:-$DATASETS_ROOT/cifar-100-python}"
    ;;
  mobilenetv2_cifar10)
    EXP_SCRIPT="code/experiments/mobilenet_cifar10.py"
    OPTIMIZER="${OPTIMIZER:-adam}"
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    LR="${LR:-5e-4}"
    MIN_LR="${MIN_LR:-5e-6}"
    GRAD_CLIP="${GRAD_CLIP:-0}"
    DATA_ROOT="${DATA_ROOT:-$DATASETS_ROOT/cifar-10-python}"
    ;;
  *)
    echo "Unknown CASE: $CASE" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$ROOT_DIR/$EXP_SCRIPT"   --method "$METHOD"   --optimizer "$OPTIMIZER"   --wd "$WD"   --seed "$SEED"   --epochs "$EPOCHS"   --batch-size "$BATCH_SIZE"   --lr "$LR"   --min-lr "$MIN_LR"   --num-workers "$NUM_WORKERS"   --grad-clip "$GRAD_CLIP"   --run-tag "$RUN_TAG"   --data-root "$DATA_ROOT"   --output-dir "$OUTPUT_DIR"   --alphadecay-root "$ALPHADECAY_ROOT"   --unbalanced-wd-every "$UNBALANCED_WD_EVERY"   --wd-min-ratio "$WD_MIN_RATIO"   --wd-max-ratio "$WD_MAX_RATIO"   --oui-window "$OUI_WINDOW"   --oui-sample-mode "$OUI_SAMPLE_MODE"   --adadecay-alpha "$ADACADECY_ALPHA"   --adadecay-eps "$ADACADECY_EPS"
