# Practitioner Guide

This folder is the quickest way to use OUIDecay in your own training loop.

## The core file

- `oui_decay_core.py`

It contains:
- `list_dynamic_modules(...)` to discover the layers that can receive dynamic WD
- `OUICollector` to measure OUI from forward activations
- `OUIDecayScheduler` to map OUI values to per-layer weight decay values
- `build_optimizer_with_module_groups(...)` to build Adam / AdamW with dynamic WD groups

## Minimal integration pattern

1. Build the model.
2. Create the optimizer with grouped parameters.
3. Register the collector hooks.
4. Run the normal training step.
5. After `optimizer.step()`, call `scheduler.step(global_step, collector.step_values())`.

The important knobs are:
- `base_wd`: the scalar WD you would use as the baseline
- `s1`, `s2`: the min/max scaling window used by OUIDecay
- `update_gap`: how many optimizer steps elapse between WD updates
- `window`: optional smoothing window over recent OUI values

## Example

See `example_usage.py` for a runnable toy example that shows the full training loop.
