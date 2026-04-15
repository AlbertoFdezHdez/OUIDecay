# Ablation Time

This folder reproduces the update-gap ablation for EfficientNet-B0 + Stanford Cars.

## What changes

- model: EfficientNet-B0
- dataset: Stanford Cars
- optimizer: Adam
- method: OUIDecay
- base WD: `5e-5`
- seed: `1`
- sweep: `update_gap = 1, 4, 16, 64, 128, 256, 512, 1024`

## Launcher

Use `experiments/launch_ablation_time.sh`.
The launcher writes outputs to `ablations/ablation_time/results/`.

## Display

Use `display/build_table.py` to rebuild the table with best validation loss and total runtime.
