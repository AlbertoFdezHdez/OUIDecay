# OUIDecay Public Release

This folder is the clean, reproducible GitHub-ready release of the OUIDecay code base used in the accompanying paper.

## Start Here

If you want to use OUIDecay in your own training code, start with:

1. [`practitioner/`](./practitioner)
   - a minimal implementation of the OUIDecay core
   - a small example showing how to plug it into a normal PyTorch training loop

If you want to reproduce the paper, go to:

2. [`main_experiments/`](./main_experiments)
   - the main CNN experiments and the scripts that rebuild the paper tables and figures
3. [`ablations/`](./ablations)
   - the update-gap, scale and overhead studies

The original project tree is preserved under [`code/`](./code) for completeness, but the release you should use is `code_ready_for_git/`.

## What OUIDecay Does

OUIDecay assigns different weight-decay values to different layers by tracking an activation-based signal called OUI. Every `t` optimizer steps, it maps the per-layer OUI scores into a WD range around the base weight decay using the `(s1, s2)` window.

## Repository Layout

- [`practitioner/`](./practitioner): copy-paste friendly core and usage example
- [`main_experiments/`](./main_experiments): main paper experiments, results and display scripts
- [`ablations/`](./ablations): timing, overhead and scale ablations
- [`code/`](./code): preserved source tree used to build the public release

## Dependencies

The code is written for standard PyTorch + torchvision training loops. The display scripts also use NumPy, Matplotlib and pickle-based result files.

No datasets are included in the repository. Each training script documents the dataset it expects and the default root directory it uses.

## Reproducibility Guide

The scripts and results are organized so that each part of the paper has a clear entry point:

- Main results: [`main_experiments/results/`](./main_experiments/results)
- Main tables and figures: [`main_experiments/display/`](./main_experiments/display)
- Update-gap ablation: [`ablations/ablation_time/`](./ablations/ablation_time)
- Overhead profiling: [`ablations/measure_overhead/`](./ablations/measure_overhead)
- Scale ablation: [`ablations/ablation_scale/`](./ablations/ablation_scale)

Each `display/` folder contains the script that rebuilds the corresponding table or figure from the saved pickles.