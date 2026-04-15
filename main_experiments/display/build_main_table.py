from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "main_experiments" / "results"
OUTPUT_DIR = ROOT_DIR / "main_experiments" / "display" / "generated_tables"
OUTPUT_PDF = OUTPUT_DIR / "VIP_final_table_4_experiments_ICONIP.pdf"

def _win_long(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    prefix = '\\?' + '\\'
    return text if text.startswith(prefix) else prefix + text

@dataclass(frozen=True)
class RowSpec:
    title: str
    model: str
    dataset: str
    optimizer: str
    wd_values: tuple[float, float]
    ada_method: str
    ada_label: str


ROW_SPECS: tuple[RowSpec, ...] = (
    RowSpec(
        title="EfficientNet-B0 + Stanford Cars",
        model="efficientnetb0",
        dataset="stanfordcars",
        optimizer="adam",
        wd_values=(1e-5, 5e-5),
        ada_method="adadecayg",
        ada_label="AdaDecayG",
    ),
    RowSpec(
        title="ResNet50 + Food101",
        model="resnet50",
        dataset="food101",
        optimizer="adamw",
        wd_values=(1e-2, 5e-2),
        ada_method="adadecay",
        ada_label="AdaDecay",
    ),
    RowSpec(
        title="DenseNet121 + CIFAR100",
        model="densenet121",
        dataset="cifar100",
        optimizer="adamw",
        wd_values=(5e-2, 1e-1),
        ada_method="adadecay",
        ada_label="AdaDecay",
    ),
    RowSpec(
        title="MobileNetV2 + CIFAR10",
        model="mobilenetv2",
        dataset="cifar10",
        optimizer="adam",
        wd_values=(1e-4, 5e-4),
        ada_method="adadecayg",
        ada_label="AdaDecayG",
    ),
)

METHODS = ("constant", "adadecay", "ouidecay")
METHOD_LABELS = {
    "constant": "WD fixed",
    "adadecay": "AdaDecay",
    "ouidecay": "OUIDecay",
}


def wd_label(weight_decay: float) -> str:
    if weight_decay < 1e-3:
        return f"{weight_decay:.0e}".replace("e-0", "e-")
    if weight_decay < 0.1:
        return f"{weight_decay:.3f}".rstrip("0").rstrip(".")
    return f"{weight_decay:.1f}".rstrip("0").rstrip(".")


def fmt(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "-"
    return f"{mean:.4f} +/- {std:.4f}"


def load_score(payload: dict) -> float | None:
    summary = payload.get("summary", {}) or {}
    value = summary.get("best_val_loss")
    if value is None:
        history = payload.get("history", {}) or {}
        val_loss = history.get("val_loss")
        if val_loss is None:
            return None
        arr = np.asarray(val_loss, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr.min())
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def extract_key(payload: dict) -> tuple[str, str, str, str, float, int] | None:
    metadata = payload.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        return None
    try:
        model = str(metadata.get("model", "")).lower()
        dataset = str(metadata.get("dataset_slug", metadata.get("dataset_name", ""))).lower().replace("-", "")
        optimizer = str(metadata.get("optimizer", "")).lower()
        method = str(metadata.get("storage_method", metadata.get("method", ""))).lower()
        wd = float(metadata.get("weight_decay"))
        seed = int(metadata.get("seed"))
    except Exception:
        return None
    return model, dataset, optimizer, method, wd, seed


def build_index(results_dir: Path) -> dict[tuple[str, str, str, str, float, int], float]:
    index: dict[tuple[str, str, str, str, float, int], float] = {}
    for file_path in sorted(results_dir.glob("*.pkl")):
        try:
            with open(_win_long(file_path), "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        key = extract_key(payload)
        if key is None:
            continue
        score = load_score(payload)
        if score is None:
            continue
        index[key] = score
    return index


def aggregate_row(
    index: dict[tuple[str, str, str, str, float, int], float],
    row: RowSpec,
    wd: float,
) -> dict[str, tuple[float, float, int]]:
    result: dict[str, tuple[float, float, int]] = {}
    for method in METHODS:
        values = []
        for seed in (1, 2, 3):
            key = (row.model, row.dataset, row.optimizer, method, float(wd), seed)
            if key in index:
                values.append(index[key])
        if values:
            arr = np.asarray(values, dtype=float)
            result[method] = (float(arr.mean()), float(arr.std(ddof=0)), int(arr.size))
        else:
            result[method] = (float("nan"), float("nan"), 0)
    return result


def render_table_page(pdf: PdfPages, index: dict[tuple[str, str, str, str, float, int], float]) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.axis("off")

    headers = ["Experiment", *[METHOD_LABELS[m] for m in METHODS]]
    rows: list[list[str]] = []
    best_cells: list[tuple[int, int]] = []

    for row in ROW_SPECS:
        for wd in row.wd_values:
            agg = aggregate_row(index, row, wd)
            finite_means = [(method, agg[method][0]) for method in METHODS if math.isfinite(agg[method][0])]
            best_method = min(finite_means, key=lambda item: (item[1], item[0]))[0] if finite_means else None
            row_label = f"{row.title}, {row.optimizer.upper()}, wd={wd_label(wd)}"
            row_values = [row_label]
            for method in METHODS:
                mean, std, _n = agg[method]
                row_values.append(fmt(mean, std))
                if best_method is not None and method == best_method:
                    best_cells.append((len(rows) + 1, METHODS.index(method) + 1))
            rows.append(row_values)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.49, 0.17, 0.17, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#2d3748")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif c == 0:
            cell.set_facecolor("#f8fafc")
            cell.get_text().set_ha("left")

    for r, c in best_cells:
        if (r, c) in table.get_celld():
            table[(r, c)].get_text().set_fontweight("bold")

    fig.suptitle(
        "Final CNN comparison table: mean +/- std across seeds",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.03,
        "Rows show the 8 selected experiment settings. The AdaDecay column is AdaDecayG for EfficientNet and MobileNet, and AdaDecay for ResNet and DenseNet. Bold marks the best mean in each row.",
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_repro_text() -> str:
    lines = [
        "Reproducibility notes",
        "",
        "Metric: mean +/- std of best_val_loss across seeds 1, 2, and 3, aggregated from descargas/results/*.pkl by the metadata fields model, dataset, optimizer, storage_method, weight_decay, and seed.",
        "The runs were selected from the final CNN sweep and use the exact training code found in the corresponding python files, with the same augmentation, schedule, and clipping logic as the executed launchers.",
        "",
        "EfficientNet-B0 + Stanford Cars (Adam, wd=1e-5 and 5e-5, AdaDecayG): the script trains for 100 epochs with batch size 64, base lr 8e-4, min lr 1e-5, 8 workers, label smoothing 0.05, and grad clipping at 1.0. The training transform uses RandomResizedCrop(224, scale=0.7-1.0, ratio=0.85-1.15), RandomHorizontalFlip, ColorJitter, RandAugment, and RandomErasing; validation uses Resize(256) and CenterCrop(224). Learning rate follows warmup plus cosine decay. No mixup or cutmix is used.",
        "",
        "ResNet50 + Food101 (AdamW, wd=1e-2 and 5e-2, AdaDecay): the script trains for 50 epochs with batch size 128, base lr 3e-4, min lr 3e-5, and no explicit grad clipping. Training uses RandomResizedCrop(224, scale=0.6-1.0), RandomHorizontalFlip, ColorJitter, and standard normalization; validation uses Resize(256) and CenterCrop(224). The optimizer follows the same warmup plus cosine schedule used in the code, and the launcher keeps the dataset local under the repository datasets directory.",
        "",
        "DenseNet121 + CIFAR100 (AdamW, wd=5e-2 and 1e-1, AdaDecay): the script trains for 100 epochs with batch size 256, base lr 5e-4, min lr 5e-6, fused AdamW enabled in auto mode, and no grad clipping. Training uses RandomCrop(32, padding=4), RandomHorizontalFlip, RandAugment(num_ops=2, magnitude=9), and RandomErasing; validation uses plain tensor conversion and normalization. The learning rate again uses warmup and cosine decay.",
        "",
        "MobileNetV2 + CIFAR10 (Adam, wd=1e-4 and 5e-4, AdaDecayG): the script trains for 100 epochs with batch size 256, base lr 5e-4, min lr 5e-6, fused Adam enabled in the launcher configuration, and no grad clipping. Training uses the same CIFAR-style augmentation as DenseNet, namely RandomCrop(32, padding=4), RandomHorizontalFlip, RandAugment(num_ops=2, magnitude=9), and RandomErasing; validation is normalization only. The learning-rate schedule is warmup plus cosine decay.",
        "",
        "Columns:",
        "- WD fixed = constant weight decay baseline.",
        "- AdaDecay = AdaDecay or AdaDecayG, depending on the row above.",
        "- OUIDecay = the original OUIDecay scheduler.",
    ]
    return "\n".join(lines)


def render_repro_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(15.5, 10.0))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        make_repro_text(),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        linespacing=1.45,
    )
    fig.suptitle(
        "VIP final table: reproducibility details",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index(RESULTS_DIR)
    with PdfPages(_win_long(OUTPUT_PDF)) as pdf:
        render_table_page(pdf, index)
        render_repro_page(pdf)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()