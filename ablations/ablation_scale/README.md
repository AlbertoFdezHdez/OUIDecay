# Ablation Scale

This folder reproduces the `(s1, s2)` sweep for OUIDecay.

## What changes

- model: ResNet50
- dataset: Food101
- optimizer: AdamW
- method: OUIDecay
- base WD: `5e-2`
- seed: `1, 2, 3`
- standard window: `(0.6666, 5.0)`
- alternative windows: `(0.6666, 3.0)`, `(0.3333, 3.0)`, `(0.3333, 5.0)`, `(0.5, 2.0)`

## Launcher

Use `experiments/launch_scale.sh`.

## Display

Use `display/build_table.py` to regenerate the scale table.
