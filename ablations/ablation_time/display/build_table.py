from __future__ import annotations

import csv
import math
import pickle
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / 'ablations' / 'ablation_time' / 'results'
OUTPUT_PDF = ROOT_DIR / 'ablations' / 'ablation_time' / 'display' / 'ablation_time_table.pdf'
OUTPUT_CSV = ROOT_DIR / 'ablations' / 'ablation_time' / 'display' / 'ablation_time_table.csv'
EXPECTED_GAPS = [1, 4, 16, 64, 128, 256, 512, 1024]

def _win_long(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    prefix = '\\?' + '\\'
    return text if text.startswith(prefix) else prefix + text

def load_payload(path: Path) -> dict:
    with open(_win_long(path), 'rb') as handle:
        return pickle.load(handle)


def extract_best_val_loss(payload: dict) -> float:
    summary = payload.get('summary', {}) or {}
    value = summary.get('best_val_loss')
    if value is not None and math.isfinite(float(value)):
        return float(value)
    history = payload.get('history', {}) or {}
    vals = history.get('val_loss', [])
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return float('nan')
    return min(vals)


def extract_duration(payload: dict) -> float:
    summary = payload.get('summary', {}) or {}
    value = summary.get('duration_s')
    return float(value) if value is not None else float('nan')


def pick_latest(paths: list[Path]) -> Path | None:
    best = None
    best_ts = -1.0
    for path in paths:
        try:
            payload = load_payload(path)
        except Exception:
            continue
        ts = float(payload.get('created_at_unix') or path.stat().st_mtime)
        if ts >= best_ts:
            best = path
            best_ts = ts
    return best


def fmt(value: float) -> str:
    return 'MISSING' if not math.isfinite(value) else f'{value:.4f}'


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for gap in EXPECTED_GAPS:
        candidates = []
        for path in RESULTS_DIR.glob('*.pkl'):
            try:
                payload = load_payload(path)
            except Exception:
                continue
            md = payload.get('metadata', {})
            cfg = payload.get('config', {})
            if md.get('benchmark') != 'efficientnetb0_stanfordcars':
                continue
            if md.get('optimizer') != 'adam':
                continue
            if md.get('method') != 'ouidecay':
                continue
            if float(md.get('weight_decay', -1)) != 5e-5:
                continue
            if int(md.get('seed', -1)) != 1:
                continue
            if int(cfg.get('unbalanced_wd_every', -1)) != gap:
                continue
            candidates.append(path)
        path = pick_latest(candidates)
        if path is None:
            rows.append([str(gap), 'MISSING', 'MISSING'])
            continue
        payload = load_payload(path)
        rows.append([
            str(gap),
            fmt(extract_best_val_loss(payload)),
            f"{extract_duration(payload):.2f}",
        ])

    with open(_win_long(OUTPUT_CSV), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['update_gap', 'best_val_loss', 'duration_s'])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['update_gap', 'best_val_loss', 'duration_s'],
        cellLoc='center',
        colLoc='center',
        loc='center',
        colWidths=[0.18, 0.36, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor('#1f2937')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif c == 0:
            cell.set_facecolor('#f8fafc')
    fig.suptitle('EfficientNet-B0 + Stanford Cars: update-gap ablation', fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02, 'Single-seed sweep. Missing rows are reported explicitly.', ha='center', va='bottom', fontsize=9)
    with PdfPages(_win_long(OUTPUT_PDF)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUTPUT_PDF}')
    print(f'Wrote {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
