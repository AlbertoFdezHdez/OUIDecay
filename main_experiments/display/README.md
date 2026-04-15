# Display for Main Experiments

- `build_main_table.py` rebuilds the final table used in the paper and writes it under `generated_tables/`.
- `plot_wd_profiles.py` rebuilds the WD snapshot figures and writes them under `generated_figures/`.

Both scripts read from `main_experiments/results/`.

Recommended order:
1. regenerate the main table
2. regenerate the WD profile figures
3. compare the outputs with the copied PDFs in `generated_tables/` and `generated_figures/`
