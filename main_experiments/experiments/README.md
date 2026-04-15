# Main Experiments

This folder contains the generic launcher used to reproduce the main CNN experiments in the paper.

## What it covers

- EfficientNet-B0 + Stanford Cars
- ResNet50 + Food101
- DenseNet121 + CIFAR100
- MobileNetV2 + CIFAR10

For each family, the launcher supports:
- fixed WD baseline
- AdaDecay or AdaDecayG, depending on the family
- OUIDecay

## Generic launcher

Use `launch_main_generic.sh` and set the case-specific environment variables documented inside the script.

The launcher writes outputs to `main_experiments/results/` by default.

## Display

Use `main_experiments/display/build_main_table.py` and `main_experiments/display/plot_wd_profiles.py` to regenerate the paper table and the WD profile figures.
