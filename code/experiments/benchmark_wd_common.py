from __future__ import annotations

import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_ROOT = SCRIPT_PATH.parent
CODE_ROOT = EXPERIMENTS_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_ALPHADECAY_ROOT = CODE_ROOT / "vendor" / "alphadecay_llama"

DATASETS_ROOT = Path(
    os.environ.get("DATASETS_ROOT", str(PROJECT_ROOT / "datasets"))
).expanduser()
DESCARGAS_ROOT = PROJECT_ROOT / "descargas"
RESULTS_ROOT = DESCARGAS_ROOT / "results"
JOB_LOGS_ROOT = DESCARGAS_ROOT / "logs" / "jobs"

DATASET_DIR_NAMES = {
    "cifar100": "cifar-100-python",
    "cifar10": "cifar10",
    "food101": "food-101",
    "stanfordcars": "stanford-cars",
    "wikitext": "wikitext",
}

METHOD_STORAGE_NAMES = {
    "wd0": "wd0",
    "uniform": "constant",
    "alphadecay": "alphadecay",
    "adadecay": "adadecay",
    "adadecayg": "adadecayg",
    "ouidecay": "ouidecay",
    "ouidecay2": "ouidecay2",
    "ouidecay3": "ouidecay3",
    "ouidecay4a": "ouidecay4a",
    "ouidecay4b": "ouidecay4b",
}


if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(raw_path: str | Path | None, default: Path) -> Path:
    if raw_path in (None, ""):
        return default
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_dataset_dir(name: str) -> Path:
    return DATASETS_ROOT / DATASET_DIR_NAMES.get(name, name)


def default_cache_dir(name: str) -> Path:
    if name == "huggingface":
        return DATASETS_ROOT
    return default_dataset_dir(name)


def default_output_dir() -> Path:
    return RESULTS_ROOT


def default_hf_cache_dir(run_group: str) -> Path:
    return PROJECT_ROOT / "datasets" / "hf_cache" / run_group


def add_alphadecay_to_path(alphadecay_root: Path) -> None:
    if not alphadecay_root.exists():
        raise FileNotFoundError(f"AlphaDecay path not found: {alphadecay_root}")
    path_str = str(alphadecay_root.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slugify_text(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "unknown"


def wd_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "")


def storage_method_name(method: str) -> str:
    return METHOD_STORAGE_NAMES.get(method, method)


def dataset_slug(dataset_name: str) -> str:
    return slugify_text(dataset_name)


def run_pickle_name(
    *,
    model: str,
    dataset_name: str,
    optimizer: str,
    method: str,
    weight_decay: float,
    seed: int,
    run_tag: str = "",
) -> str:
    parts = [
        f"model={slugify_text(model)}",
        f"dataset={dataset_slug(dataset_name)}",
        f"optimizer={slugify_text(optimizer)}",
        f"method={slugify_text(storage_method_name(method))}",
        f"wd={wd_tag(weight_decay)}",
        f"seed={int(seed)}",
    ]
    if run_tag:
        parts.append(f"tag={slugify_text(run_tag)}")
    return "_".join(parts) + ".pkl"


def write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def rows_to_series(rows: Sequence[Mapping[str, object]]) -> dict[str, list[object]]:
    if not rows:
        return {}
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return {key: [row.get(key) for row in rows] for key in keys}


def build_run_payload(
    *,
    kind: str,
    benchmark: str,
    model: str,
    dataset_name: str,
    optimizer: str,
    method: str,
    weight_decay: float,
    seed: int,
    output_path: Path,
    run_tag: str,
    config: Mapping[str, object],
    summary: Mapping[str, object],
    history_rows: Sequence[Mapping[str, object]],
    wd_history: Sequence[Mapping[str, object]],
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = {
        "benchmark": benchmark,
        "model": model,
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug(dataset_name),
        "optimizer": optimizer,
        "method": method,
        "storage_method": storage_method_name(method),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "run_tag": str(run_tag or ""),
        "result_file": output_path.name,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return {
        "schema_version": 1,
        "kind": kind,
        "metadata": metadata,
        "config": dict(config),
        "summary": dict(summary),
        "history_rows": [dict(row) for row in history_rows],
        "history": rows_to_series(history_rows),
        "wd_history": [dict(row) for row in wd_history],
        "created_at_unix": float(time.time()),
    }
