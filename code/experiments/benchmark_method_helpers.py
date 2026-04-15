from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from benchmark_wd_common import add_alphadecay_to_path
from torch_accel import build_adam, build_adamw
from training.generic_oui_decay import (
    AlphaDecayCompatWrapper,
    GenericOUICollector,
    build_optimizer_with_module_groups,
    build_standard_optimizer,
    current_dynamic_wd,
    list_decay_parameters,
    list_dynamic_modules,
)
from training.llama_oui_decay import (
    OUIDecay2Scheduler,
    OUIDecay3Scheduler,
    OUIDecay4AScheduler,
    OUIDecay4BScheduler,
    OUIDecayScheduler,
)


@dataclass
class MethodState:
    optimizer: torch.optim.Optimizer
    collector: GenericOUICollector | None
    scheduler: OUIDecayScheduler | None
    alphadecay: object | None
    adadecay_params: list[torch.nn.Parameter]
    adadecay_wd: float
    adadecay_alpha: float
    adadecay_eps: float
    alphadecay_group_names: list[str]
    wd_history: list[dict[str, float]]
    dynamic_module_names: list[str]
    fused_effective: bool | None


def capture_named_group_wd(
    optimizer: torch.optim.Optimizer,
    group_names: Sequence[str],
    *,
    step: int,
    extra: Mapping[str, float | int | str | None] | None = None,
) -> dict[str, float | int | str | None]:
    row: dict[str, float | int | str | None] = {"step": float(step)}
    if extra:
        row.update(dict(extra))
    for idx, name in enumerate(group_names):
        if idx >= len(optimizer.param_groups):
            break
        row[f"wd::{name}"] = float(optimizer.param_groups[idx]["weight_decay"])
    if len(optimizer.param_groups) > len(group_names):
        row["wd::other"] = float(optimizer.param_groups[-1]["weight_decay"])
    return row


def _alphadecay_group_names(
    model_wrapper: nn.Module,
    alphadecay_obj,
    layer_stats,
) -> list[str]:
    layer_name_to_tune = set(layer_stats["longname"].tolist())
    group_names: list[str] = []
    for name, module in model_wrapper.named_modules():
        if name in layer_name_to_tune:
            group_names.append(name)
            continue
        if (
            getattr(alphadecay_obj, "batchnorm", False)
            and isinstance(module, nn.BatchNorm2d)
            and name in getattr(alphadecay_obj, "bn_to_conv", {})
            and alphadecay_obj.bn_to_conv[name] in layer_name_to_tune
        ):
            group_names.append(name)
    return group_names


def build_method_state(
    *,
    model: nn.Module,
    method: str,
    optimizer_name: str,
    lr: float,
    base_wd: float,
    betas: tuple[float, float],
    device: torch.device,
    alphadecay_root: Path,
    assign_func: str,
    wd_min_ratio: float,
    wd_max_ratio: float,
    esd_metric_for_tb: str,
    pl_fitting: str,
    xmin_pos: float,
    filter_zeros: bool,
    remove_first_layer: bool,
    remove_last_layer: bool,
    batchnorm: bool,
    batchnorm_type: str,
    unbalanced_wd_every: int,
    oui_window: int,
    oui_eps: float,
    oui_sample_mode: str,
    total_updates_estimate: int,
    oui3_oui_strength: float,
    oui3_depth_strength: float,
    oui3_time_start: float,
    oui3_time_end: float,
    oui3_scale_min: float,
    oui3_scale_max: float,
    oui4a_rank_strength: float,
    oui4a_scale_min: float,
    oui4a_scale_max: float,
    oui4b_drift_strength: float,
    oui4b_ema_decay: float,
    oui4b_scale_min: float,
    oui4b_scale_max: float,
    adadecay_alpha: float,
    adadecay_eps: float,
    adamw_fused_mode: str = "auto",
    seed: int = 0,
) -> MethodState:
    method = str(method).lower()
    optimizer_name = str(optimizer_name).lower()

    if method == "wd0":
        optimizer = build_standard_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=0.0,
            betas=betas,
            adamw_fused_mode=adamw_fused_mode,
            device_type=device.type,
        )
        return MethodState(
            optimizer=optimizer,
            collector=None,
            scheduler=None,
            alphadecay=None,
            adadecay_params=[],
            adadecay_wd=0.0,
            adadecay_alpha=float(adadecay_alpha),
            adadecay_eps=float(adadecay_eps),
            alphadecay_group_names=[],
            wd_history=[{"step": 0.0, "weight_decay": 0.0}],
            dynamic_module_names=[],
            fused_effective=getattr(optimizer, "_adamw_fused_effective", None),
        )

    if method == "uniform":
        optimizer = build_standard_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=base_wd,
            betas=betas,
            adamw_fused_mode=adamw_fused_mode,
            device_type=device.type,
        )
        return MethodState(
            optimizer=optimizer,
            collector=None,
            scheduler=None,
            alphadecay=None,
            adadecay_params=[],
            adadecay_wd=0.0,
            adadecay_alpha=float(adadecay_alpha),
            adadecay_eps=float(adadecay_eps),
            alphadecay_group_names=[],
            wd_history=[{"step": 0.0, "weight_decay": float(base_wd)}],
            dynamic_module_names=[],
            fused_effective=getattr(optimizer, "_adamw_fused_effective", None),
        )

    if method in {"adadecay", "adadecayg"}:
        optimizer = build_standard_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=0.0,
            betas=betas,
            adamw_fused_mode=adamw_fused_mode,
            device_type=device.type,
        )
        return MethodState(
            optimizer=optimizer,
            collector=None,
            scheduler=None,
            alphadecay=None,
            adadecay_params=list_decay_parameters(model),
            adadecay_wd=float(base_wd),
            adadecay_alpha=float(adadecay_alpha),
            adadecay_eps=float(adadecay_eps),
            alphadecay_group_names=[],
            wd_history=[
                {
                    "step": 0.0,
                    "weight_decay": float(base_wd),
                    "alpha": float(adadecay_alpha),
                    "eps": float(adadecay_eps),
                }
            ],
            dynamic_module_names=[],
            fused_effective=getattr(optimizer, "_adamw_fused_effective", None),
        )

    if method in {"ouidecay", "ouidecay2", "ouidecay3", "ouidecay4a", "ouidecay4b"}:
        module_map = list_dynamic_modules(
            model,
            remove_first_layer=remove_first_layer,
            remove_last_layer=remove_last_layer,
        )
        if not module_map:
            raise RuntimeError("OUIDecay found no eligible Conv2d/Linear/Conv1D modules.")

        optimizer, group_info = build_optimizer_with_module_groups(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            base_wd=base_wd,
            dynamic_module_names=list(module_map.keys()),
            betas=betas,
            adamw_fused_mode=adamw_fused_mode,
            device_type=device.type,
        )
        active_module_names = list(group_info.module_group_indices.keys())
        active_module_map = {name: module_map[name] for name in active_module_names}
        collector = GenericOUICollector(
            active_module_map,
            sample_mode=oui_sample_mode,
            seed=seed,
        )
        if method == "ouidecay2":
            scheduler = OUIDecay2Scheduler(
                optimizer,
                group_info.module_group_indices,
                base_wd=base_wd,
                update_gap=unbalanced_wd_every,
                s1=wd_min_ratio,
                s2=wd_max_ratio,
                eps=oui_eps,
                window=oui_window,
            )
        elif method == "ouidecay3":
            scheduler = OUIDecay3Scheduler(
                optimizer,
                group_info.module_group_indices,
                base_wd=base_wd,
                update_gap=unbalanced_wd_every,
                s1=wd_min_ratio,
                s2=wd_max_ratio,
                eps=oui_eps,
                window=oui_window,
                total_updates=total_updates_estimate,
                oui_strength=oui3_oui_strength,
                depth_strength=oui3_depth_strength,
                time_start=oui3_time_start,
                time_end=oui3_time_end,
                scale_min=oui3_scale_min,
                scale_max=oui3_scale_max,
            )
        elif method == "ouidecay4a":
            scheduler = OUIDecay4AScheduler(
                optimizer,
                group_info.module_group_indices,
                base_wd=base_wd,
                update_gap=unbalanced_wd_every,
                s1=wd_min_ratio,
                s2=wd_max_ratio,
                eps=oui_eps,
                window=oui_window,
                rank_strength=oui4a_rank_strength,
                scale_min=oui4a_scale_min,
                scale_max=oui4a_scale_max,
            )
        elif method == "ouidecay4b":
            scheduler = OUIDecay4BScheduler(
                optimizer,
                group_info.module_group_indices,
                base_wd=base_wd,
                update_gap=unbalanced_wd_every,
                s1=wd_min_ratio,
                s2=wd_max_ratio,
                eps=oui_eps,
                window=oui_window,
                drift_strength=oui4b_drift_strength,
                ema_decay=oui4b_ema_decay,
                scale_min=oui4b_scale_min,
                scale_max=oui4b_scale_max,
            )
        else:
            scheduler = OUIDecayScheduler(
                optimizer,
                group_info.module_group_indices,
                base_wd=base_wd,
                update_gap=unbalanced_wd_every,
                s1=wd_min_ratio,
                s2=wd_max_ratio,
                eps=oui_eps,
                window=oui_window,
            )
        return MethodState(
            optimizer=optimizer,
            collector=collector,
            scheduler=scheduler,
            alphadecay=None,
            adadecay_params=[],
            adadecay_wd=0.0,
            adadecay_alpha=float(adadecay_alpha),
            adadecay_eps=float(adadecay_eps),
            alphadecay_group_names=[],
            wd_history=[dict(row) for row in scheduler.history],
            dynamic_module_names=active_module_names,
            fused_effective=getattr(optimizer, "_adamw_fused_effective", None),
        )

    if method == "alphadecay":
        add_alphadecay_to_path(alphadecay_root)
        from WeightDecayUnbalance import modulewise_AlphaDecay

        model_wrapper = AlphaDecayCompatWrapper(model)
        alphadecay = modulewise_AlphaDecay(
            model_wrapper,
            use_modulewise_wd=True,
            EVALS_THRESH=0.00001,
            bins=100,
            conv_norm=0.5,
            pl_fitting=pl_fitting,
            xmin_pos=xmin_pos,
            filter_zeros=filter_zeros,
            remove_first_layer=remove_first_layer,
            remove_last_layer=remove_last_layer,
            eigs_thresh=50,
            esd_metric_for_tb=esd_metric_for_tb,
            assign_func=assign_func,
            wd_min_ratio=wd_min_ratio,
            wd_max_ratio=wd_max_ratio,
            batchnorm=batchnorm,
            batchnorm_type=batchnorm_type,
        )
        param_groups, layer_count, layer_stats = alphadecay.build_optimizer_param_group(
            untuned_wd=base_wd,
            initialize=True,
        )
        if int(layer_count) <= 0 or layer_stats.empty:
            raise RuntimeError("AlphaDecay found no eligible layers for this architecture.")

        if optimizer_name == "adam":
            optimizer, fused_effective = build_adam(param_groups, lr=lr, betas=betas)
        elif optimizer_name == "adamw":
            optimizer, fused_effective = build_adamw(
                param_groups,
                device=device,
                lr=lr,
                betas=betas,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        group_names = _alphadecay_group_names(model_wrapper, alphadecay, layer_stats)
        wd_history = [
            capture_named_group_wd(
                optimizer,
                group_names,
                step=0,
                extra={"layer_count": int(layer_count)},
            )
        ]
        return MethodState(
            optimizer=optimizer,
            collector=None,
            scheduler=None,
            alphadecay=alphadecay,
            adadecay_params=[],
            adadecay_wd=0.0,
            adadecay_alpha=float(adadecay_alpha),
            adadecay_eps=float(adadecay_eps),
            alphadecay_group_names=group_names,
            wd_history=wd_history,
            dynamic_module_names=[],
            fused_effective=fused_effective,
        )

    raise ValueError(f"Unsupported method: {method}")


def maybe_update_ouidecay(
    *,
    method_state: MethodState,
    update_step: int,
) -> dict[str, float] | None:
    if method_state.collector is None or method_state.scheduler is None:
        return None
    oui_values = method_state.collector.step_values()
    return method_state.scheduler.step(update_step, oui_values)


def maybe_update_alphadecay(
    *,
    method_state: MethodState,
    untuned_wd: float,
    update_step: int,
) -> dict[str, float | int | str | None] | None:
    if method_state.alphadecay is None:
        return None
    alpha_summary = method_state.alphadecay.step(
        method_state.optimizer,
        untuned_wd=untuned_wd,
        step_count=update_step,
        rank0=True,
    )
    return capture_named_group_wd(
        method_state.optimizer,
        method_state.alphadecay_group_names,
        step=update_step,
        extra=alpha_summary,
    )


@torch.no_grad()
def _apply_adadecay_impl(method_state: MethodState) -> None:
    if not method_state.adadecay_params:
        return

    wd0 = float(method_state.adadecay_wd)
    alpha = float(method_state.adadecay_alpha)
    eps = float(method_state.adadecay_eps)

    for param in method_state.adadecay_params:
        grad = param.grad
        if grad is None:
            continue
        abs_grad = grad.abs()
        mu = abs_grad.mean()
        std = abs_grad.std(unbiased=False)
        denom = torch.clamp(std, min=eps)
        g_tilde = (abs_grad - mu) / denom
        theta = 2.0 / (1.0 + torch.exp(-alpha * g_tilde))
        grad.add_(wd0 * theta * param)


@torch.no_grad()
def maybe_apply_adadecay(method_state: MethodState) -> None:
    _apply_adadecay_impl(method_state)


@torch.no_grad()
def maybe_apply_adadecayg(method_state: MethodState) -> None:
    _apply_adadecay_impl(method_state)


def current_ouidecay_wd(method_state: MethodState) -> Dict[str, float]:
    if method_state.scheduler is None:
        return {}
    return current_dynamic_wd(
        method_state.optimizer,
        method_state.scheduler.module_group_indices,
    )
