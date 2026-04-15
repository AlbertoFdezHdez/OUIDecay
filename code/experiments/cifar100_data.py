from __future__ import annotations

from pathlib import Path

from torchvision import datasets

from benchmark_wd_common import default_dataset_dir, resolve_repo_path


def is_cifar100_dir(path: Path) -> bool:
    return all((path / name).exists() for name in ("train", "test", "meta"))


def _candidate_torchvision_roots(raw_path: str | None) -> list[Path]:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("cifar100"))
    candidates: list[Path] = []
    if requested_root.name == "cifar-100-python":
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


def build_cifar100_datasets(
    raw_path: str | None,
    train_transform,
    val_transform,
    *,
    download_if_missing: bool = True,
):
    candidates = _candidate_torchvision_roots(raw_path)

    for torchvision_root in candidates:
        dataset_dir = torchvision_root / "cifar-100-python"
        if not is_cifar100_dir(dataset_dir):
            continue
        train_set = datasets.CIFAR100(
            root=str(torchvision_root),
            train=True,
            download=False,
            transform=train_transform,
        )
        val_set = datasets.CIFAR100(
            root=str(torchvision_root),
            train=False,
            download=False,
            transform=val_transform,
        )
        return dataset_dir, torchvision_root, train_set, val_set, 100, False

    if not download_if_missing:
        checked = ", ".join(str(path / "cifar-100-python") for path in candidates)
        raise FileNotFoundError(
            "CIFAR100 not found. Expected the extracted dataset under one of: "
            f"{checked}"
        )

    download_root = candidates[0]
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        train_set = datasets.CIFAR100(
            root=str(download_root),
            train=True,
            download=True,
            transform=train_transform,
        )
        val_set = datasets.CIFAR100(
            root=str(download_root),
            train=False,
            download=True,
            transform=val_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "CIFAR100 download/setup failed. "
            f"Tried to place the dataset under {download_root / 'cifar-100-python'}. "
            "Check network access, disk space, and torchvision dataset support."
        ) from exc

    dataset_dir = download_root / "cifar-100-python"
    if not is_cifar100_dir(dataset_dir):
        raise RuntimeError(
            "CIFAR100 download finished but the extracted directory layout is incomplete. "
            f"Expected {dataset_dir} to contain 'train', 'test', and 'meta'."
        )
    return dataset_dir, download_root, train_set, val_set, 100, True
