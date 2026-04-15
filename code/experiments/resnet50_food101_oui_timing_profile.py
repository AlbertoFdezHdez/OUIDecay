from __future__ import annotations

import argparse
import math
import statistics as stats
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from benchmark_method_helpers import build_method_state, maybe_update_ouidecay
from benchmark_wd_common import (
    DEFAULT_ALPHADECAY_ROOT,
    build_run_payload,
    default_output_dir,
    resolve_repo_path,
    run_pickle_name,
    set_seed,
    write_pickle,
)
from food101_data import build_food101_datasets
from torch_accel import bf16_autocast, use_cuda_bf16


BENCHMARK_NAME = 'resnet50_food101_oui_timing_profile'
MODEL_NAME = 'resnet50'
DATASET_NAME = 'food101'
OPTIMIZER_NAME = 'adam'
BETAS = (0.95, 0.95)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(stats.mean(values)), float(stats.pstdev(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ResNet50 + Food101 OUI timing profiler.')
    parser.add_argument('--method', type=str, default='ouidecay', choices=['ouidecay'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--wd', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=str(default_output_dir()))
    parser.add_argument('--run-tag', type=str, default='profile_timing')
    parser.add_argument('--alphadecay-root', type=str, default=str(DEFAULT_ALPHADECAY_ROOT))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--profile-update-gap', type=int, default=10)
    parser.add_argument('--assign-func', type=str, default='tb_linear_map')
    parser.add_argument('--wd-min-ratio', type=float, default=0.6666)
    parser.add_argument('--wd-max-ratio', type=float, default=5.0)
    parser.add_argument('--esd-metric-for-tb', type=str, default='alpha')
    parser.add_argument('--pl-fitting', type=str, default='median', choices=['median', 'goodness-of-fit', 'fix-finger'])
    parser.add_argument('--xmin-pos', type=float, default=2.0)
    parser.add_argument('--filter-zeros', action='store_true')
    parser.add_argument('--remove-first-layer', action='store_true', default=True)
    parser.add_argument('--remove-last-layer', action='store_true', default=True)
    parser.add_argument('--batchnorm', action='store_true', default=True)
    parser.add_argument('--batchnorm-type', type=str, default='name')
    parser.add_argument('--oui-window', type=int, default=5)
    parser.add_argument('--oui-eps', type=float, default=1e-8)
    parser.add_argument('--oui-sample-mode', type=str, default='random', choices=['random', 'first'])
    parser.add_argument('--oui3-oui-strength', type=float, default=1.0)
    parser.add_argument('--oui3-depth-strength', type=float, default=0.6)
    parser.add_argument('--oui3-time-start', type=float, default=1.3)
    parser.add_argument('--oui3-time-end', type=float, default=0.25)
    parser.add_argument('--oui3-scale-min', type=float, default=0.2)
    parser.add_argument('--oui3-scale-max', type=float, default=4.0)
    parser.add_argument('--oui4a-rank-strength', type=float, default=1.0)
    parser.add_argument('--oui4a-scale-min', type=float, default=0.2)
    parser.add_argument('--oui4a-scale-max', type=float, default=4.0)
    parser.add_argument('--oui4b-drift-strength', type=float, default=1.0)
    parser.add_argument('--oui4b-ema-decay', type=float, default=0.95)
    parser.add_argument('--oui4b-scale-min', type=float, default=0.2)
    parser.add_argument('--oui4b-scale-max', type=float, default=4.0)
    parser.add_argument('--adadecay-alpha', type=float, default=4.0)
    parser.add_argument('--adadecay-eps', type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer_name = str(args.optimizer).lower()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = use_cuda_bf16(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_dir, torchvision_root, train_set, val_set, num_classes, downloaded = build_food101_datasets(
        args.data_root,
        tf_train,
        tf_val,
        download_if_missing=True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.to(device)

    epochs = int(args.epochs)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    max_val_batches = len(val_set) if hasattr(val_set, '__len__') else 0
    method_state = build_method_state(
        model=model,
        method='ouidecay',
        optimizer_name=optimizer_name,
        lr=args.lr,
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
        unbalanced_wd_every=args.profile_update_gap,
        oui_window=args.oui_window,
        oui_eps=args.oui_eps,
        oui_sample_mode=args.oui_sample_mode,
        total_updates_estimate=total_steps,
        oui3_oui_strength=args.oui3_oui_strength,
        oui3_depth_strength=args.oui3_depth_strength,
        oui3_time_start=args.oui3_time_start,
        oui3_time_end=args.oui3_time_end,
        oui3_scale_min=args.oui3_scale_min,
        oui3_scale_max=args.oui3_scale_max,
        oui4a_rank_strength=args.oui4a_rank_strength,
        oui4a_scale_min=args.oui4a_scale_min,
        oui4a_scale_max=args.oui4a_scale_max,
        oui4b_drift_strength=args.oui4b_drift_strength,
        oui4b_ema_decay=args.oui4b_ema_decay,
        oui4b_scale_min=args.oui4b_scale_min,
        oui4b_scale_max=args.oui4b_scale_max,
        adadecay_alpha=args.adadecay_alpha,
        adadecay_eps=args.adadecay_eps,
        seed=args.seed,
    )
    optimizer = method_state.optimizer

    warmup_steps = max(100, int(0.05 * total_steps))
    iter_rows: list[dict[str, float | int | bool]] = []
    global_step = 0

    print('Food-101 dataset dir:', dataset_dir)
    print('Food-101 torchvision root:', torchvision_root)
    print('Food-101 auto-downloaded:', downloaded)
    print('Num classes:', num_classes)
    print('Precision mode:', 'bf16_autocast' if use_bf16 else 'fp32')
    print('Optimizer:', optimizer_name)
    print('Method:', args.method)
    print('Profile update gap:', int(args.profile_update_gap))
    print('Fused AdamW:', method_state.fused_effective)
    print(f'Run config | epochs={epochs} | train_batches={steps_per_epoch} | wd={args.wd:g} | workers={8}')

    t_start = time.perf_counter()

    try:
        if epochs != 1:
            raise ValueError('This profiler is intended to run exactly 1 epoch.')

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        train_batches = 0
        epoch_t0 = time.perf_counter()
        loader_iter = iter(train_loader)

        pbar = tqdm(range(steps_per_epoch), desc='Epoch 001/001', leave=True, dynamic_ncols=True)
        for batch_idx in pbar:
            iter_t0 = time.perf_counter()
            x, y = next(loader_iter)
            fetch_t1 = time.perf_counter()

            x = x.to(device)
            y = y.to(device)

            if global_step < warmup_steps:
                lr = args.lr * (global_step + 1) / max(1, warmup_steps)
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(max(float(progress), 0.0), 1.0)
                lr = float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress)))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if method_state.collector is not None:
                method_state.collector.reset_step()

            optimizer.zero_grad(set_to_none=True)
            with bf16_autocast(device, use_bf16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            did_update = bool(args.profile_update_gap > 0 and global_step % args.profile_update_gap == 0)
            oui_update_sec = 0.0
            if did_update:
                _sync(device)
                oui_t0 = time.perf_counter()
                wd_values = maybe_update_ouidecay(method_state=method_state, update_step=global_step)
                _sync(device)
                oui_update_sec = time.perf_counter() - oui_t0
                if wd_values is not None and method_state.scheduler is not None:
                    method_state.wd_history = list(method_state.scheduler.history)

            _sync(device)
            iter_end = time.perf_counter()

            fetch_sec = fetch_t1 - iter_t0
            compute_sec = iter_end - fetch_t1
            iter_wall_sec = iter_end - iter_t0

            acc = accuracy(logits, y)
            running_loss += float(loss.item())
            running_acc += float(acc)
            train_batches += 1

            iter_rows.append(
                {
                    'iter_idx': int(batch_idx + 1),
                    'global_step': int(global_step),
                    'did_oui_update': bool(did_update),
                    'fetch_sec': float(fetch_sec),
                    'compute_sec': float(compute_sec),
                    'iter_wall_sec': float(iter_wall_sec),
                    'oui_update_sec': float(oui_update_sec),
                    'loss': float(loss.item()),
                    'acc': float(acc),
                    'lr': float(lr),
                }
            )

            pbar.set_postfix(
                loss=f'{running_loss / train_batches:.4f}',
                acc=f'{running_acc / train_batches:.4f}',
                lr=f'{lr:.2e}',
                update='yes' if did_update else 'no',
                iter=f'{iter_wall_sec:.3f}s',
            )

        pbar.close()
        train_sec = time.perf_counter() - epoch_t0
    finally:
        if method_state.collector is not None:
            method_state.collector.close()

    update_rows = [row for row in iter_rows if bool(row['did_oui_update'])]
    non_update_rows = [row for row in iter_rows if not bool(row['did_oui_update'])]
    update_iter_secs = [float(row['iter_wall_sec']) for row in update_rows]
    non_update_iter_secs = [float(row['iter_wall_sec']) for row in non_update_rows]
    oui_update_secs = [float(row['oui_update_sec']) for row in update_rows]

    update_iter_mean, update_iter_std = _mean_std(update_iter_secs)
    non_update_iter_mean, non_update_iter_std = _mean_std(non_update_iter_secs)
    oui_update_mean, oui_update_std = _mean_std(oui_update_secs)

    output_dir = resolve_repo_path(args.output_dir, default_output_dir())
    output_path = output_dir / run_pickle_name(
        model=MODEL_NAME,
        dataset_name=DATASET_NAME,
        optimizer=optimizer_name,
        method='ouidecay',
        weight_decay=args.wd,
        seed=args.seed,
        run_tag=args.run_tag,
    )

    summary = {
        'epochs': int(epochs),
        'steps_per_epoch': int(steps_per_epoch),
        'updates': int(global_step),
        'profile_update_gap': int(args.profile_update_gap),
        'n_update_iters': int(len(update_rows)),
        'n_non_update_iters': int(len(non_update_rows)),
        'mean_iter_wall_sec_update': float(update_iter_mean),
        'std_iter_wall_sec_update': float(update_iter_std),
        'mean_iter_wall_sec_no_update': float(non_update_iter_mean),
        'std_iter_wall_sec_no_update': float(non_update_iter_std),
        'mean_oui_update_sec': float(oui_update_mean),
        'std_oui_update_sec': float(oui_update_std),
        'train_sec': float(train_sec),
        'duration_s': float(time.perf_counter() - t_start),
        'best_val_loss': None,
        'final_val_loss': None,
        'final_train_loss': float(running_loss / max(1, train_batches)),
        'final_train_acc': float(running_acc / max(1, train_batches)),
    }
    config = dict(vars(args))
    config.update({'optimizer': optimizer_name, 'betas': BETAS})
    payload = build_run_payload(
        kind='food101_oui_timing_profile',
        benchmark=BENCHMARK_NAME,
        model=MODEL_NAME,
        dataset_name=DATASET_NAME,
        optimizer=optimizer_name,
        method='ouidecay',
        weight_decay=args.wd,
        seed=args.seed,
        output_path=output_path,
        run_tag=args.run_tag,
        config=config,
        summary=summary,
        history_rows=iter_rows,
        wd_history=method_state.scheduler.history if method_state.scheduler is not None else method_state.wd_history,
        extra_metadata={
            'num_classes': int(num_classes),
            'batch_size': int(args.batch_size),
            'profile_update_gap': int(args.profile_update_gap),
            'steps_per_epoch': int(steps_per_epoch),
            'n_update_iters': int(len(update_rows)),
        },
    )
    write_pickle(output_path, payload)
    print(
        f"[RUN_DONE] benchmark={BENCHMARK_NAME} method=ouidecay optimizer={optimizer_name} "
        f"wd={args.wd:g} seed={args.seed} epochs={epochs} steps_per_epoch={steps_per_epoch} "
        f"updates={global_step} n_update_iters={len(update_rows)} "
        f"mean_iter_update={update_iter_mean:.4f}s mean_iter_no_update={non_update_iter_mean:.4f}s "
        f"mean_oui_update={oui_update_mean:.4f}s duration_s={summary['duration_s']:.1f} "
        f"output={output_path}"
    )


if __name__ == '__main__':
    main()