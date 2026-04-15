from __future__ import annotations

from pathlib import Path

from torchvision import datasets

from benchmark_wd_common import default_dataset_dir, resolve_repo_path


def is_cifar10_dir(path: Path) -> bool:
    return all((path / name).exists() for name in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch", "batches.meta"))


def _candidate_torchvision_roots(raw_path: str | None) -> list[Path]:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("cifar10"))
    candidates: list[Path] = []
    if requested_root.name == "cifar-10-batches-py":
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


def build_cifar10_datasets(
    raw_path: str | None,
    train_transform,
    val_transform,
    *,
    download_if_missing: bool = True,
):
    candidates = _candidate_torchvision_roots(raw_path)

    for torchvision_root in candidates:
        dataset_dir = torchvision_root / "cifar-10-batches-py"
        if not is_cifar10_dir(dataset_dir):
            continue
        train_set = datasets.CIFAR10(
            root=str(torchvision_root),
            train=True,
            download=False,
            transform=train_transform,
        )
        val_set = datasets.CIFAR10(
            root=str(torchvision_root),
            train=False,
            download=False,
            transform=val_transform,
        )
        return dataset_dir, torchvision_root, train_set, val_set, 10, False

    if not download_if_missing:
        checked = ", ".join(str(path / "cifar-10-batches-py") for path in candidates)
        raise FileNotFoundError(
            "CIFAR10 not found. Expected the extracted dataset under one of: "
            f"{checked}"
        )

    download_root = candidates[0]
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        train_set = datasets.CIFAR10(
            root=str(download_root),
            train=True,
            download=True,
            transform=train_transform,
        )
        val_set = datasets.CIFAR10(
            root=str(download_root),
            train=False,
            download=True,
            transform=val_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "CIFAR10 download/setup failed. "
            f"Tried to place the dataset under {download_root / 'cifar-10-batches-py'}. "
            "Check network access, disk space, and torchvision dataset support."
        ) from exc

    dataset_dir = download_root / "cifar-10-batches-py"
    if not is_cifar10_dir(dataset_dir):
        raise RuntimeError(
            "CIFAR10 download finished but the extracted directory layout is incomplete. "
            f"Expected {dataset_dir} to contain the standard CIFAR10 batch files."
        )
    return dataset_dir, download_root, train_set, val_set, 10, True
