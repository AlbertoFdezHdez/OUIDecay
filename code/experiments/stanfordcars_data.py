from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets
from torchvision.datasets import ImageFolder

from benchmark_wd_common import default_dataset_dir, resolve_repo_path


GITHUB_ZIP_URL = (
    "https://codeload.github.com/jhpohovey/StanfordCars-Dataset/zip/refs/heads/main"
)


def is_stanfordcars_imagefolder_root(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def is_stanfordcars_torchvision_root(path: Path) -> bool:
    stanford_cars_dir = path / "stanford_cars"
    return (
        (stanford_cars_dir / "cars_train").is_dir()
        and (stanford_cars_dir / "cars_test").is_dir()
        and (stanford_cars_dir / "devkit").is_dir()
    )


def _candidate_roots(raw_path: str | None) -> list[Path]:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("stanfordcars"))
    candidates = [requested_root]
    if requested_root.name != "stanford-cars":
        candidates.append(requested_root / "stanford-cars")
    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _download_stanfordcars_repo(download_root: Path) -> Path:
    download_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stanfordcars_", dir=str(download_root)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "stanfordcars.zip"
        try:
            urllib.request.urlretrieve(GITHUB_ZIP_URL, archive_path)
        except Exception as exc:
            raise RuntimeError(
                "Stanford Cars download failed while fetching the public GitHub mirror. "
                "The original torchvision host is known to be unavailable upstream. "
                f"Tried URL: {GITHUB_ZIP_URL}"
            ) from exc
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp_path)
        except Exception as exc:
            raise RuntimeError(
                "Stanford Cars archive was downloaded but could not be extracted cleanly."
            ) from exc

        extracted_root = tmp_path / "StanfordCars-Dataset-main" / "stanford_cars"
        if not is_stanfordcars_torchvision_root(tmp_path / "StanfordCars-Dataset-main"):
            raise RuntimeError(
                "Stanford Cars mirror download succeeded, but the extracted repository layout "
                "is not the expected one for torchvision.datasets.StanfordCars."
            )

        target_root = download_root / "stanford_cars"
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(extracted_root), str(target_root))
    return target_root


def build_stanfordcars_datasets(
    raw_path: str | None,
    train_transform,
    val_transform,
    *,
    download_if_missing: bool = True,
):
    candidates = _candidate_roots(raw_path)

    for candidate in candidates:
        if is_stanfordcars_imagefolder_root(candidate):
            train_dataset = ImageFolder(candidate / "train", transform=train_transform)
            val_dataset = ImageFolder(candidate / "test", transform=val_transform)
            num_classes = len(train_dataset.classes)
            return candidate, train_dataset, val_dataset, num_classes, False

    for candidate in candidates:
        torchvision_root = candidate if candidate.name != "stanford-cars" else candidate.parent
        if is_stanfordcars_torchvision_root(torchvision_root):
            try:
                train_dataset = datasets.StanfordCars(
                    root=str(torchvision_root),
                    split="train",
                    download=False,
                    transform=train_transform,
                )
                val_dataset = datasets.StanfordCars(
                    root=str(torchvision_root),
                    split="test",
                    download=False,
                    transform=val_transform,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Stanford Cars files were found, but torchvision could not read them. "
                    "Check whether scipy is installed and the dataset layout is complete."
                ) from exc
            return torchvision_root / "stanford_cars", train_dataset, val_dataset, len(train_dataset.classes), False

    if not download_if_missing:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Stanford Cars not found. Expected either an ImageFolder layout under "
            f"{checked} or a torchvision-compatible 'stanford_cars' directory."
        )

    download_root = candidates[0] if candidates[0].name != "stanford-cars" else candidates[0].parent
    target_root = _download_stanfordcars_repo(download_root)
    try:
        train_dataset = datasets.StanfordCars(
            root=str(download_root),
            split="train",
            download=False,
            transform=train_transform,
        )
        val_dataset = datasets.StanfordCars(
            root=str(download_root),
            split="test",
            download=False,
            transform=val_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "Stanford Cars mirror was downloaded, but torchvision still could not open it. "
            "This usually means scipy is missing or the local dataset files are corrupted."
        ) from exc
    return target_root, train_dataset, val_dataset, len(train_dataset.classes), True
