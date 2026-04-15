from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / 'ablations' / 'measure_overhead' / 'results'
OUTPUT_PDF = ROOT_DIR / 'ablations' / 'measure_overhead' / 'display' / 'measure_overhead_table.pdf'
OUTPUT_CSV = ROOT_DIR / 'ablations' / 'measure_overhead' / 'display' / 'measure_overhead_table.csv'

def _win_long(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    prefix = '\\?' + '\\'
    return text if text.startswith(prefix) else prefix + text
NETWORKS = [
    ('EfficientNet-B0 + Stanford Cars', 'efficientnetb0', 'stanfordcars', 'adam', 5e-5),
    ('ResNet50 + Food101', 'resnet50', 'food101', 'adam', 5e-2),
    ('DenseNet121 + CIFAR100', 'densenet121', 'cifar100', 'adam', 5e-2),
    ('MobileNetV2 + CIFAR10', 'mobilenetv2', 'cifar10', 'adam', 5e-4),
]


def load_payload(path: Path) -> dict:
    with open(_win_long(path), 'rb') as handle:
        return pickle.load(handle)


def fmt_ms(value: float, std: float) -> str:
    return f'{value * 1000:.2f} +/- {std * 1000:.2f} ms'


def main() -> None:
    rows = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for title, model, dataset, optimizer, wd in NETWORKS:
        match = None
        for path in RESULTS_DIR.glob('*.pkl'):
            try:
                payload = load_payload(path)
            except Exception:
                continue
            md = payload.get('metadata', {})
            if md.get('benchmark') not in {f'{model}_{dataset}', f'{model}_{dataset}_oui_timing_profile'}:
                continue
            if md.get('model') != model or md.get('dataset_slug') != dataset:
                continue
            if md.get('optimizer') != optimizer:
                continue
            if float(md.get('weight_decay', -1)) != wd:
                continue
            if md.get('run_tag') != 'profile_timing':
                continue
            match = payload
            break
        if match is None:
            rows.append([title, 'MISSING', 'MISSING', 'MISSING'])
            continue
        summary = match.get('summary', {}) or {}
        iter_mean = float(summary['mean_iter_wall_sec_no_update'])
        iter_std = float(summary['std_iter_wall_sec_no_update'])
        update_mean = float(summary['mean_oui_update_sec'])
        update_std = float(summary['std_oui_update_sec'])
        pct = (update_mean / iter_mean) * 100.0 if math.isfinite(iter_mean) and iter_mean > 0 else float('nan')
        rows.append([title, fmt_ms(iter_mean, iter_std), fmt_ms(update_mean, update_std), f'{pct:.3f}%'])

    with open(_win_long(OUTPUT_CSV), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['network', 'iter_wall_ms', 'oui_update_ms', 'pct_overall'])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['network', 'iter_wall_ms', 'oui_update_ms', 'pct_overall'],
        cellLoc='center',
        colLoc='center',
        loc='center',
        colWidths=[0.34, 0.23, 0.23, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor('#1f2937')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif c == 0:
            cell.set_facecolor('#f8fafc')
    fig.suptitle('OUI update overhead', fontsize=15, fontweight='bold', y=0.98)
    with PdfPages(_win_long(OUTPUT_PDF)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUTPUT_PDF}')
    print(f'Wrote {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
