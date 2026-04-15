from __future__ import annotations

import argparse
import csv
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "main_experiments" / "results"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "main_experiments" / "display" / "generated_figures"

SNAPSHOT_LABELS = ("Inicio", "Medio", "Final")

METHOD_COLORS = {
    "uniform": "#7f7f7f",
    "adadecay": "#1f77b4",
    "adadecayg": "#17becf",
    "ouidecay": "#ff7f0e",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str
    preferred_run_tags: tuple[str, ...]


@dataclass(frozen=True)
class NetworkSpec:
    key: str
    title: str
    benchmark: str
    optimizer: str
    wd: float
    seed: int
    methods: tuple[MethodSpec, ...]
    output_name: str


NETWORK_SPECS: tuple[NetworkSpec, ...] = (
    NetworkSpec(
        key="efficientnetb0_stanfordcars",
        title="EfficientNet-B0 + Stanford Cars",
        benchmark="efficientnetb0_stanfordcars",
        optimizer="adam",
        wd=5e-5,
        seed=1,
        methods=(
            MethodSpec("uniform", "WD fijo", ("carsalphaoui", "ouidecay_ablation", "article_seed_repeats", "")),
            MethodSpec("adadecayg", "AdaDecayG", ("adadecayg_ouidecay2_cnn4fam", "cnn_sota_final_missing", "")),
            MethodSpec("ouidecay", "OUIDecay", ("carsalphaoui", "ouidecay_ablation", "ouidecay_ablation_gaps_3seed", "article_seed_repeats", "")),
        ),
        output_name="cnn_sota_wd_snapshots_efficientnetb0_stanfordcars.pdf",
    ),
    NetworkSpec(
        key="resnet50_food101",
        title="ResNet50 + Food101",
        benchmark="resnet50_food101",
        optimizer="adamw",
        wd=5e-2,
        seed=1,
        methods=(
            MethodSpec("uniform", "WD fijo", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
            MethodSpec("adadecay", "AdaDecay", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
            MethodSpec("ouidecay", "OUIDecay", ("cnn_sota_4bench_fullswap", "ouidecay_ablation", "ouidecay_ablation_gaps_3seed", "cnn_sota_final_missing", "")),
        ),
        output_name="cnn_sota_wd_snapshots_resnet50_food101.pdf",
    ),
    NetworkSpec(
        key="densenet121_cifar100",
        title="DenseNet121 + CIFAR100",
        benchmark="densenet121_cifar100",
        optimizer="adamw",
        wd=1e-1,
        seed=1,
        methods=(
            MethodSpec("uniform", "WD fijo", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
            MethodSpec("adadecay", "AdaDecay", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
            MethodSpec("ouidecay", "OUIDecay", ("cnn_sota_4bench_fullswap", "cnn_sota_ouidecay_newscale", "cnn_sota_final_missing", "")),
        ),
        output_name="cnn_sota_wd_snapshots_densenet121_cifar100.pdf",
    ),
    NetworkSpec(
        key="mobilenetv2_cifar10",
        title="MobileNetV2 + CIFAR10",
        benchmark="mobilenetv2_cifar10",
        optimizer="adam",
        wd=5e-4,
        seed=1,
        methods=(
            MethodSpec("uniform", "WD fijo", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
            MethodSpec("adadecayg", "AdaDecayG", ("adadecayg_ouidecay2_cnn4fam", "cnn_sota_final_missing", "")),
            MethodSpec("ouidecay", "OUIDecay", ("cnn_sota_4bench_fullswap", "cnn_sota_final_missing", "")),
        ),
        output_name="cnn_sota_wd_snapshots_mobilenetv2_cifar10.pdf",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WD snapshots (start / middle / final) for the CNN suite."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the experiment PKLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PDFs will be written.",
    )
    parser.add_argument(
        "--networks",
        type=str,
        default=",".join(spec.key for spec in NETWORK_SPECS),
        help="Comma-separated list of network keys to plot.",
    )
    return parser.parse_args()


def normalize_tag(tag: object) -> str:
    if tag in (None, ""):
        return ""
    return str(tag)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_close(a: float, b: float, *, atol: float = 1e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def load_payloads(results_dir: Path, spec: NetworkSpec) -> dict[str, dict[str, object]]:
    candidates: dict[str, tuple[int, float, dict[str, object]]] = {}
    for path in sorted(results_dir.glob("*.pkl")):
        with open(_win_long(path), "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            continue
        md = payload.get("metadata", {})
        if md.get("benchmark") != spec.benchmark:
            continue
        if md.get("optimizer") != spec.optimizer:
            continue
        if int(md.get("seed", -1)) != spec.seed:
            continue
        if not is_close(safe_float(md.get("weight_decay")), spec.wd):
            continue

        method = str(md.get("method", "")).lower()
        method_spec = next((m for m in spec.methods if m.method == method), None)
        if method_spec is None:
            continue

        run_tag = normalize_tag(md.get("run_tag"))
        try:
            priority = method_spec.preferred_run_tags.index(run_tag)
        except ValueError:
            priority = len(method_spec.preferred_run_tags)
        candidate = (priority, -path.stat().st_mtime, payload)
        previous = candidates.get(method)
        if previous is None or candidate[:2] < previous[:2]:
            candidates[method] = candidate

    payloads = {method: candidate[2] for method, candidate in candidates.items()}
    missing = [method_spec.method for method_spec in spec.methods if method_spec.method not in payloads]
    if missing:
        raise RuntimeError(
            f"Missing PKLs for {spec.key}: {missing}. Available methods: {sorted(payloads)}"
        )
    return payloads


def scalar_wd_from_payload(payload: dict[str, object]) -> float:
    md = payload.get("metadata", {})
    return float(md.get("weight_decay", float("nan")))


def extract_dynamic_snapshot_stats(payload: dict[str, object]) -> list[dict[str, float]]:
    wd_history = payload.get("wd_history", [])
    if not wd_history:
        scalar = scalar_wd_from_payload(payload)
        return [
            {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
            {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
            {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
        ]

    indices = [0, len(wd_history) // 2, len(wd_history) - 1]
    snapshots: list[dict[str, float]] = []
    for idx in indices:
        row = wd_history[idx]
        values = [float(v) for key, v in row.items() if key.startswith("wd::") and math.isfinite(float(v))]
        if values:
            snapshots.append(
                {
                    "step": float(row.get("step", idx)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "count": float(len(values)),
                }
            )
        else:
            scalar = scalar_wd_from_payload(payload)
            snapshots.append(
                {
                    "step": float(row.get("step", idx)),
                    "mean": scalar,
                    "std": 0.0,
                    "count": 1.0,
                }
            )
    return snapshots


def method_snapshot_stats(method: str, payload: dict[str, object]) -> list[dict[str, float]]:
    if method == "ouidecay":
        return extract_dynamic_snapshot_stats(payload)

    scalar = scalar_wd_from_payload(payload)
    return [
        {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
        {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
        {"step": 0.0, "mean": scalar, "std": 0.0, "count": 1.0},
    ]


def set_sci_axis(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def plot_network(spec: NetworkSpec, payloads: dict[str, dict[str, object]], output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / spec.output_name

    snapshot_data = {
        method_spec.method: method_snapshot_stats(method_spec.method, payloads[method_spec.method])
        for method_spec in spec.methods
    }
    reference_method = next((m.method for m in spec.methods if m.method == "ouidecay"), spec.methods[0].method)
    reference_steps = [entry["step"] for entry in snapshot_data[reference_method]]

    records: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    fig.suptitle(
        f"{spec.title} | wd={spec.wd:g} | seed={spec.seed}",
        fontsize=14,
        fontweight="bold",
    )

    bar_methods = [m for m in spec.methods]
    x = np.arange(len(bar_methods))
    colors = [METHOD_COLORS.get(m.method, "#999999") for m in bar_methods]

    for col_idx, ax in enumerate(axes):
        heights = [snapshot_data[m.method][col_idx]["mean"] for m in bar_methods]
        errors = [snapshot_data[m.method][col_idx]["std"] for m in bar_methods]
        bars = ax.bar(x, heights, yerr=errors, capsize=4, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        for bar, height, err in zip(bars, heights, errors):
            ax.annotate(
                f"{height:.2e}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
                rotation=0,
            )
        ax.set_title(f"{SNAPSHOT_LABELS[col_idx]} | step={int(reference_steps[col_idx])}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.label for m in bar_methods], rotation=0, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        set_sci_axis(ax)
        if col_idx == 0:
            ax.set_ylabel("Mean WD", fontsize=11)
        ax.set_ylim(0, max(heights) * 1.35 if heights else 1.0)

        for method_spec, height, err in zip(bar_methods, heights, errors):
            records.append(
                {
                    "network": spec.key,
                    "title": spec.title,
                    "seed": spec.seed,
                    "wd": spec.wd,
                    "snapshot": SNAPSHOT_LABELS[col_idx],
                    "step": int(reference_steps[col_idx]),
                    "method": method_spec.label,
                    "method_name": method_spec.method,
                    "mean_wd": height,
                    "std_wd": err,
                    "n_groups": snapshot_data[method_spec.method][col_idx]["count"],
                }
            )

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected = {item.strip() for item in args.networks.split(",") if item.strip()}
    specs = [spec for spec in NETWORK_SPECS if spec.key in selected]
    if not specs:
        raise RuntimeError(f"No matching networks selected from: {sorted(selected)}")

    all_rows: list[dict[str, object]] = []
    for spec in specs:
        payloads = load_payloads(args.results_dir, spec)
        rows = plot_network(spec, payloads, args.output_dir)
        all_rows.extend(rows)
        print(f"Wrote {args.output_dir / spec.output_name}")

    summary_csv = args.output_dir / "cnn_sota_wd_snapshots_summary.csv"
    write_csv(summary_csv, all_rows)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()


def _win_long(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    prefix = '\\?' + '\\'
    return text if text.startswith(prefix) else prefix + text
