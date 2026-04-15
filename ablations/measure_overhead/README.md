# Overhead Measurement

This folder reproduces the timing profiling used to measure the overhead of OUI updates.

## What is measured

For each family we report:
- mean time of a full iteration
- mean time spent in the OUI/WD update block
- the percentage that the update block represents relative to the full iteration

## Covered families

- EfficientNet-B0 + Stanford Cars
- ResNet50 + Food101
- DenseNet121 + CIFAR100
- MobileNetV2 + CIFAR10

## Launcher

Use `experiments/launch_overhead.sh`.

## Display

Use `display/build_table.py` to rebuild the summary table.
