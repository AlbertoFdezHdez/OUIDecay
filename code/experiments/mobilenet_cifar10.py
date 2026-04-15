from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from benchmark_method_helpers import (
    build_method_state,
    maybe_apply_adadecay,
    maybe_apply_adadecayg,
    maybe_update_alphadecay,
    maybe_update_ouidecay,
)
from benchmark_wd_common import (
    DEFAULT_ALPHADECAY_ROOT,
    build_run_payload,
    default_output_dir,
    run_pickle_name,
    set_seed,
    write_pickle,
)
from cifar10_data import build_cifar10_datasets
from torch_accel import bf16_autocast, use_cuda_bf16


BENCHMARK_NAME = "mobilenetv2_cifar10"
MODEL_NAME = "mobilenetv2"
DATASET_NAME = "cifar10"
OPTIMIZER_NAME = "adamw"
BETAS = (0.95, 0.95)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileNetV2 + CIFAR10 WD-method benchmark.")
    parser.add_argument("--method", type=str, required=True, choices=["wd0", "uniform", "alphadecay", "adadecay", "adadecayg", "ouidecay", "ouidecay2"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--alphadecay-root", type=str, default=str(DEFAULT_ALPHADECAY_ROOT))
    parser.add_argument("--adamw-fused-mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--unbalanced-wd-every", type=int, default=500)
    parser.add_argument("--assign-func", type=str, default="tb_linear_map")
    parser.add_argument("--wd-min-ratio", type=float, default=0.6666)
    parser.add_argument("--wd-max-ratio", type=float, default=5.0)
    parser.add_argument("--esd-metric-for-tb", type=str, default="alpha")
    parser.add_argument("--pl-fitting", type=str, default="median", choices=["median", "goodness-of-fit", "fix-finger"])
    parser.add_argument("--xmin-pos", type=float, default=2.0)
    parser.add_argument("--filter-zeros", action="store_true")
    parser.add_argument("--remove-first-layer", action="store_true", default=True)
    parser.add_argument("--remove-last-layer", action="store_true", default=True)
    parser.add_argument("--batchnorm", action="store_true", default=True)
    parser.add_argument("--batchnorm-type", type=str, default="name")
    parser.add_argument("--oui-window", type=int, default=5)
    parser.add_argument("--oui-eps", type=float, default=1e-8)
    parser.add_argument("--oui-sample-mode", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--adadecay-alpha", type=float, default=4.0)
    parser.add_argument("--adadecay-eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OPTIMIZER_NAME = str(args.optimizer).lower()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_cuda_bf16(device)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0), value="random"),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_dir, torchvision_root, train_set, val_set, num_classes, downloaded = build_cifar10_datasets(
        args.data_root,
        tf_train,
        tf_val,
        download_if_missing=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = models.mobilenet_v2(weights=None)
    model.features[0][0].stride = (1, 1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)

    base_lr = 5e-4
    min_lr = 5e-6
    warmup_steps = 500
    epochs = int(args.epochs)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    max_val_batches = len(val_loader)
    method_state = build_method_state(
        model=model,
        method=args.method,
        optimizer_name=OPTIMIZER_NAME,
        lr=base_lr,
        base_wd=args.wd,
        betas=BETAS,
        device=device,
        alphadecay_root=Path(args.alphadecay_root),
        assign_func=args.assign_func,
        wd_min_ratio=args.wd_min_ratio,
        wd_max_ratio=args.wd_max_ratio,
        esd_metric_for_tb=args.esd_metric_for_tb,
        pl_fitting=args.pl_fitting,
        xmin_pos=args.xmin_pos,
        filter_zeros=args.filter_zeros,
        remove_first_layer=args.remove_first_layer,
        remove_last_layer=args.remove_last_layer,
        batchnorm=args.batchnorm,
        batchnorm_type=args.batchnorm_type,
        unbalanced_wd_every=args.unbalanced_wd_every,
        oui_window=args.oui_window,
        oui_eps=args.oui_eps,
        oui_sample_mode=args.oui_sample_mode,
        total_updates_estimate=total_steps,
        oui3_oui_strength=1.0,
        oui3_depth_strength=0.6,
        oui3_time_start=1.3,
        oui3_time_end=0.25,
        oui3_scale_min=0.2,
        oui3_scale_max=4.0,
        oui4a_rank_strength=1.0,
        oui4a_scale_min=0.2,
        oui4a_scale_max=4.0,
        oui4b_drift_strength=1.0,
        oui4b_ema_decay=0.95,
        oui4b_scale_min=0.2,
        oui4b_scale_max=4.0,
        adadecay_alpha=args.adadecay_alpha,
        adadecay_eps=args.adadecay_eps,
        adamw_fused_mode=args.adamw_fused_mode,
        seed=args.seed,
    )
    optimizer = method_state.optimizer

    history_rows: list[dict[str, float | int]] = []
    global_step = 0
    t_start = time.time()

    print("CIFAR10 dataset dir:", dataset_dir)
    print("CIFAR10 torchvision root:", torchvision_root)
    print("CIFAR10 auto-downloaded:", downloaded)
    print("Precision mode:", "bf16_autocast" if use_bf16 else "fp32")
    print("Optimizer:", OPTIMIZER_NAME)
    print("Method:", args.method)
    print("Fused AdamW:", method_state.fused_effective)
    print(
        f"Run config | epochs={epochs} | train_batches={steps_per_epoch} | "
        f"val_batches={max_val_batches} | wd={args.wd:g}"
    )

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            train_batches = 0
            train_t0 = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=True, dynamic_ncols=True)
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)

                if method_state.collector is not None:
                    method_state.collector.reset_step()

                if global_step < warmup_steps:
                    lr = base_lr * (global_step + 1) / max(1, warmup_steps)
                else:
                    t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    t = min(max(float(t), 0.0), 1.0)
                    lr = float(min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t)))

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with bf16_autocast(device, use_bf16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
                loss.backward()
                if args.method == "adadecay":
                    maybe_apply_adadecay(method_state)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if args.method == "adadecayg":
                    maybe_apply_adadecayg(method_state)
                optimizer.step()
                global_step += 1

                if method_state.scheduler is not None:
                    maybe_update_ouidecay(method_state=method_state, update_step=global_step)
                elif method_state.alphadecay is not None and global_step % args.unbalanced_wd_every == 0:
                    row = maybe_update_alphadecay(method_state=method_state, untuned_wd=args.wd, update_step=global_step)
                    if row is not None:
                        method_state.wd_history.append(row)

                acc = accuracy(logits, y)
                running_loss += float(loss.item())
                running_acc += float(acc)
                train_batches += 1
                pbar.set_postfix(
                    loss=f"{running_loss / train_batches:.4f}",
                    acc=f"{running_acc / train_batches:.4f}",
                    lr=f"{lr:.2e}",
                )

            train_loss = running_loss / max(1, train_batches)
            train_acc = running_acc / max(1, train_batches)
            pbar.close()
            train_sec = time.time() - train_t0

            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_batches = 0
            val_t0 = time.time()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    with bf16_autocast(device, use_bf16):
                        logits = model(x)
                        loss = F.cross_entropy(logits, y)
                    val_loss += float(loss.item())
                    val_acc += float(accuracy(logits, y))
                    val_batches += 1

            val_loss /= max(1, val_batches)
            val_acc /= max(1, val_batches)
            val_sec = time.time() - val_t0
            epoch_sec = train_sec + val_sec
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "lr": float(lr),
                    "train_sec": float(train_sec),
                    "val_sec": float(val_sec),
                    "epoch_sec": float(epoch_sec),
                }
            )
            tqdm.write(
                f"Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.4f} | "
                f"val {val_loss:.4f}/{val_acc:.4f} | batches train={train_batches} val={val_batches} | "
                f"time train={train_sec:.1f}s val={val_sec:.1f}s epoch={epoch_sec:.1f}s"
            )
    finally:
        if method_state.collector is not None:
            method_state.collector.close()

    best_row = min(history_rows, key=lambda row: float(row["val_loss"]))
    final_row = history_rows[-1]
    output_path = Path(args.output_dir) / run_pickle_name(
        model=MODEL_NAME,
        dataset_name=DATASET_NAME,
        optimizer=OPTIMIZER_NAME,
        method=args.method,
        weight_decay=args.wd,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    summary = {
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_acc": float(best_row["val_acc"]),
        "final_val_loss": float(final_row["val_loss"]),
        "final_val_acc": float(final_row["val_acc"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_train_acc": float(final_row["train_acc"]),
        "epochs": int(epochs),
        "updates": int(global_step),
        "duration_s": float(time.time() - t_start),
    }
    config = dict(vars(args))
    config.update(
        {
            "optimizer": OPTIMIZER_NAME,
            "betas": BETAS,
            "base_lr": base_lr,
            "min_lr": min_lr,
            "warmup_steps": int(warmup_steps),
        }
    )
    payload = build_run_payload(
        kind="cifar10_classification_run",
        benchmark=BENCHMARK_NAME,
        model=MODEL_NAME,
        dataset_name=DATASET_NAME,
        optimizer=OPTIMIZER_NAME,
        method=args.method,
        weight_decay=args.wd,
        seed=args.seed,
        output_path=output_path,
        run_tag=args.run_tag,
        config=config,
        summary=summary,
        history_rows=history_rows,
        wd_history=method_state.scheduler.history if method_state.scheduler is not None else method_state.wd_history,
        extra_metadata={
            "num_classes": int(num_classes),
            "batch_size": int(args.batch_size),
            "cifar10_dataset_dir": str(dataset_dir),
            "cifar10_torchvision_root": str(torchvision_root),
            "cifar10_auto_downloaded": bool(downloaded),
            "precision_mode": "bf16_autocast" if use_bf16 else "fp32",
        },
    )
    write_pickle(output_path, payload)
    print(
        f"[RUN_DONE] benchmark={BENCHMARK_NAME} method={args.method} optimizer={OPTIMIZER_NAME} "
        f"wd={args.wd:g} seed={args.seed} epochs={epochs} best_val_loss={summary['best_val_loss']:.4f} "
        f"final_val_loss={summary['final_val_loss']:.4f} duration_s={summary['duration_s']:.1f} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
