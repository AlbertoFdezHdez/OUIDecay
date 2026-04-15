from __future__ import annotations

import inspect
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except Exception:  # pragma: no cover - optional dependency.
    HFConv1D = None


@dataclass(frozen=True)
class OUIComputationStats:
    batch_size: int
    feature_dim: int
    half_batch: int
    activation_shape: tuple[int, ...]
    binary_matrix_bytes: int
    positives_bytes: int
    minority_bytes: int
    estimated_working_set_bytes: int


@dataclass(frozen=True)
class GenericOptimizerGroupInfo:
    module_group_indices: dict[str, int]
    other_decay_group_index: int | None
    no_decay_group_index: int | None


def flatten_activations_per_sample(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError(f'Expected activations with at least 2 dims, got {tuple(activations.shape)}')
    return activations.reshape(activations.shape[0], -1)


@torch.no_grad()
def compute_new_oui_with_stats(activations: torch.Tensor) -> tuple[float, OUIComputationStats]:
    matrix = flatten_activations_per_sample(activations)
    batch_size = matrix.shape[0]
    feature_dim = matrix.shape[1]
    half_batch = batch_size // 2
    activation_shape = tuple(activations.shape)

    stats = OUIComputationStats(
        batch_size=batch_size,
        feature_dim=feature_dim,
        half_batch=half_batch,
        activation_shape=activation_shape,
        binary_matrix_bytes=matrix.numel(),
        positives_bytes=feature_dim * 8,
        minority_bytes=feature_dim * 4,
        estimated_working_set_bytes=matrix.numel() + feature_dim * 12,
    )

    if batch_size < 2 or half_batch == 0:
        return 0.0, stats

    positives = (matrix > 0).sum(dim=0)
    minority = torch.minimum(positives, batch_size - positives).to(torch.float32)
    oui = (minority / float(half_batch)).mean()
    return float(oui.item()), stats


@torch.no_grad()
def compute_new_oui_from_activations(activations: torch.Tensor) -> float:
    oui, _ = compute_new_oui_with_stats(activations)
    return oui


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model


def _eligible_module_types() -> tuple[type[nn.Module], ...]:
    base_types: list[type[nn.Module]] = [nn.Conv2d, nn.Linear]
    if HFConv1D is not None:
        base_types.append(HFConv1D)
    return tuple(base_types)


def _is_decay_parameter(full_name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad:
        return False
    if full_name.endswith('.bias'):
        return False
    if param.ndim <= 1:
        return False
    return True


def list_dynamic_modules(
    model: nn.Module,
    *,
    remove_first_layer: bool = False,
    remove_last_layer: bool = False,
) -> 'OrderedDict[str, nn.Module]':
    mapping: 'OrderedDict[str, nn.Module]' = OrderedDict()
    for name, module in _unwrap_model(model).named_modules():
        if not name:
            continue
        if not isinstance(module, _eligible_module_types()):
            continue
        weight = getattr(module, 'weight', None)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            continue
        if not weight.requires_grad or weight.ndim <= 1:
            continue
        mapping[name] = module

    names = list(mapping.keys())
    if remove_first_layer and names:
        mapping.pop(names[0], None)
        names = list(mapping.keys())
    if remove_last_layer and names:
        mapping.pop(names[-1], None)
    return mapping


def _resolve_adamw_fused_kwargs(adamw_fused_mode: str, device_type: str) -> tuple[dict, bool]:
    mode = str(adamw_fused_mode).lower()
    if mode not in {'auto', 'on', 'off'}:
        raise ValueError(f'Unsupported AdamW fused mode: {adamw_fused_mode}')

    fused_supported = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    if mode == 'on':
        if not fused_supported:
            raise RuntimeError("AdamW fused mode requested, but this torch build has no 'fused' argument.")
        if device_type != 'cuda':
            raise RuntimeError('AdamW fused mode requested, but device_type is not CUDA.')
        return {'fused': True}, True
    if mode == 'off':
        if fused_supported:
            return {'fused': False}, False
        return {}, False
    if fused_supported and device_type == 'cuda':
        return {'fused': True}, True
    if fused_supported:
        return {'fused': False}, False
    return {}, False


def build_standard_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    adamw_fused_mode: str = 'auto',
    device_type: str = 'cpu',
) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for full_name, param in _unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if _is_decay_parameter(full_name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups: list[dict[str, object]] = []
    if decay_params:
        param_groups.append({'params': decay_params, 'weight_decay': float(weight_decay)})
    if no_decay_params:
        param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas)
        setattr(optimizer, '_adamw_fused_effective', None)
        return optimizer
    if optimizer_name == 'adamw':
        extra_kwargs, fused_effective = _resolve_adamw_fused_kwargs(adamw_fused_mode, device_type)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra_kwargs)
        setattr(optimizer, '_adamw_fused_effective', bool(fused_effective))
        return optimizer
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')


def build_optimizer_with_module_groups(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    base_wd: float,
    dynamic_module_names: Sequence[str],
    betas: tuple[float, float],
    adamw_fused_mode: str = 'auto',
    device_type: str = 'cpu',
) -> tuple[torch.optim.Optimizer, GenericOptimizerGroupInfo]:
    module_map = list_dynamic_modules(model)
    missing = [name for name in dynamic_module_names if name not in module_map]
    if missing:
        raise ValueError(f'Unknown dynamic module names: {missing}')

    param_groups: list[MutableMapping[str, object]] = []
    assigned_ids: set[int] = set()
    module_group_indices: dict[str, int] = {}

    for module_name in dynamic_module_names:
        module = module_map[module_name]
        params: list[torch.nn.Parameter] = []
        for local_name, param in module.named_parameters(recurse=False):
            full_name = f'{module_name}.{local_name}'
            if id(param) in assigned_ids or not _is_decay_parameter(full_name, param):
                continue
            params.append(param)
            assigned_ids.add(id(param))
        if not params:
            continue
        module_group_indices[module_name] = len(param_groups)
        param_groups.append(
            {
                'params': params,
                'lr': lr,
                'weight_decay': float(base_wd),
                'group_name': module_name,
                'dynamic_wd': True,
            }
        )

    other_decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for full_name, param in _unwrap_model(model).named_parameters():
        if not param.requires_grad or id(param) in assigned_ids:
            continue
        if _is_decay_parameter(full_name, param):
            other_decay_params.append(param)
        else:
            no_decay_params.append(param)

    other_decay_group_index = None
    if other_decay_params:
        other_decay_group_index = len(param_groups)
        param_groups.append(
            {
                'params': other_decay_params,
                'lr': lr,
                'weight_decay': float(base_wd),
                'group_name': 'other_decay',
                'dynamic_wd': False,
            }
        )

    no_decay_group_index = None
    if no_decay_params:
        no_decay_group_index = len(param_groups)
        param_groups.append(
            {
                'params': no_decay_params,
                'lr': lr,
                'weight_decay': 0.0,
                'group_name': 'no_decay',
                'dynamic_wd': False,
            }
        )

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas)
        setattr(optimizer, '_adamw_fused_effective', None)
    elif optimizer_name == 'adamw':
        extra_kwargs, fused_effective = _resolve_adamw_fused_kwargs(adamw_fused_mode, device_type)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra_kwargs)
        setattr(optimizer, '_adamw_fused_effective', bool(fused_effective))
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    return optimizer, GenericOptimizerGroupInfo(
        module_group_indices=module_group_indices,
        other_decay_group_index=other_decay_group_index,
        no_decay_group_index=no_decay_group_index,
    )


class OUICollector:
    def __init__(self, module_map: Mapping[str, nn.Module], *, sample_mode: str = 'random', seed: int = 0) -> None:
        if sample_mode not in {'random', 'first'}:
            raise ValueError("sample_mode must be 'random' or 'first'.")
        self.sample_mode = sample_mode
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._step_values: dict[str, float] = {}
        for module_name, module in module_map.items():
            self._handles.append(module.register_forward_hook(self._make_hook(module_name)))

    def _select_indices(self, batch_size: int, num_positions: int, device: torch.device) -> torch.Tensor:
        if self.sample_mode == 'first':
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if device.type == 'cuda':
            return torch.randint(low=0, high=num_positions, size=(batch_size,), device=device)
        return torch.randint(low=0, high=num_positions, size=(batch_size,), generator=self.rng, device=device)

    def _extract_tensor(self, output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = self._extract_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(output, Mapping):
            for item in output.values():
                tensor = self._extract_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        return None

    def _features_from_tensor(self, output: torch.Tensor) -> torch.Tensor | None:
        if output.ndim < 2:
            return None
        if output.ndim == 2:
            return output
        if output.ndim == 3:
            bsz, num_tokens, channels = output.shape
            token_idx = self._select_indices(bsz, num_tokens, output.device)
            gather_index = token_idx.view(bsz, 1, 1).expand(-1, 1, channels)
            return output.gather(dim=1, index=gather_index).squeeze(1)
        if output.ndim == 4:
            bsz, channels, height, width = output.shape
            flat = output.reshape(bsz, channels, height * width)
            pos_idx = self._select_indices(bsz, height * width, output.device)
            gather_index = pos_idx.view(bsz, 1, 1).expand(-1, channels, 1)
            return flat.gather(dim=2, index=gather_index).squeeze(2)
        return output.reshape(output.shape[0], -1)

    def _make_hook(self, module_name: str):
        @torch.no_grad()
        def hook(_module: nn.Module, _inputs, output) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None:
                return
            features = self._features_from_tensor(tensor)
            if features is None:
                return
            self._step_values[module_name] = float(compute_new_oui_from_activations(features))

        return hook

    def reset_step(self) -> None:
        self._step_values = {}

    def step_values(self) -> dict[str, float]:
        return dict(self._step_values)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class OUIDecayScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        module_group_indices: Mapping[str, int],
        base_wd: float,
        update_gap: int,
        s1: float,
        s2: float,
        eps: float = 1e-8,
        window: int = 5,
    ) -> None:
        if update_gap <= 0:
            raise ValueError('update_gap must be >= 1')
        if window <= 0:
            raise ValueError('window must be >= 1')
        self.optimizer = optimizer
        self.module_group_indices = dict(module_group_indices)
        self.base_wd = float(base_wd)
        self.update_gap = int(update_gap)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.eps = float(eps)
        self.window = int(window)
        self._queues: dict[str, deque[float]] = {name: deque(maxlen=self.window) for name in self.module_group_indices}
        self.history: list[dict[str, float]] = []
        for group_idx in self.module_group_indices.values():
            self.optimizer.param_groups[group_idx]['weight_decay'] = self.base_wd
        self._log(0, {}, self.current_wd())

    def current_wd(self) -> dict[str, float]:
        return {name: float(self.optimizer.param_groups[group_idx]['weight_decay']) for name, group_idx in self.module_group_indices.items()}

    def _smoothed_oui(self) -> dict[str, float]:
        return {name: float(sum(queue) / len(queue)) for name, queue in self._queues.items() if queue}

    def _log(self, step: int, oui_values: Mapping[str, float], wd_values: Mapping[str, float]) -> None:
        row: dict[str, float] = {'step': float(step)}
        for module_name in self.module_group_indices:
            row[f'oui::{module_name}'] = float(oui_values.get(module_name, float('nan')))
            row[f'wd::{module_name}'] = float(wd_values.get(module_name, float('nan')))
        self.history.append(row)

    def step(self, update_step: int, oui_values: Mapping[str, float]) -> dict[str, float] | None:
        for module_name in self.module_group_indices:
            if module_name in oui_values:
                self._queues[module_name].append(float(oui_values[module_name]))

        if update_step % self.update_gap != 0:
            return None

        smoothed = self._smoothed_oui()
        if not smoothed:
            return None

        oui_min = min(smoothed.values())
        oui_max = max(smoothed.values())
        wd_values: dict[str, float] = {}
        for module_name, group_idx in self.module_group_indices.items():
            oui_i = smoothed.get(module_name)
            if oui_i is None:
                wd_values[module_name] = float(self.optimizer.param_groups[group_idx]['weight_decay'])
                continue
            wd_scale = self.s1 + (self.s2 - self.s1) * ((oui_i - oui_min) / (oui_max - oui_min + self.eps))
            wd_i = self.base_wd * wd_scale
            self.optimizer.param_groups[group_idx]['weight_decay'] = float(wd_i)
            wd_values[module_name] = float(wd_i)

        self._log(update_step, smoothed, wd_values)
        return wd_values
