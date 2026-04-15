from __future__ import annotations

from pathlib import Path

from torchvision import datasets

from benchmark_wd_common import default_dataset_dir, resolve_repo_path


def is_food101_dir(path: Path) -> bool:
    return (path / "meta").exists() and (path / "images").exists()


def _candidate_torchvision_roots(raw_path: str | None) -> list[Path]:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("food101"))
    candidates: list[Path] = []
    if requested_root.name == "food-101":
        candidates.append(requested_root.parent)
        candidates.append(requested_root)
    else:
        candidates.append(requested_root)
        candidates.append(requested_root.parent)
    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def build_food101_datasets(
    raw_path: str | None,
    train_transform,
    val_transform,
    *,
    download_if_missing: bool = True,
):
    candidates = _candidate_torchvision_roots(raw_path)

    for torchvision_root in candidates:
        dataset_dir = torchvision_root / "food-101"
        if not is_food101_dir(dataset_dir):
            continue
        train_set = datasets.Food101(
            root=str(torchvision_root),
            split="train",
            download=False,
            transform=train_transform,
        )
        val_set = datasets.Food101(
            root=str(torchvision_root),
            split="test",
            download=False,
            transform=val_transform,
        )
        return dataset_dir, torchvision_root, train_set, val_set, len(train_set.classes), False

    if not download_if_missing:
        checked = ", ".join(str(path / "food-101") for path in candidates)
        raise FileNotFoundError(
            "Food-101 not found. Expected the extracted dataset under one of: "
            f"{checked}"
        )

    download_root = candidates[0]
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        train_set = datasets.Food101(
            root=str(download_root),
            split="train",
            download=True,
            transform=train_transform,
        )
        val_set = datasets.Food101(
            root=str(download_root),
            split="test",
            download=True,
            transform=val_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "Food-101 download/setup failed. "
            f"Tried to place the dataset under {download_root / 'food-101'}. "
            "Check network access, disk space, and torchvision dataset support."
        ) from exc

    dataset_dir = download_root / "food-101"
    if not is_food101_dir(dataset_dir):
        raise RuntimeError(
            "Food-101 download finished but the extracted directory layout is incomplete. "
            f"Expected {dataset_dir} to contain 'meta' and 'images'."
        )
    return dataset_dir, download_root, train_set, val_set, len(train_set.classes), True
