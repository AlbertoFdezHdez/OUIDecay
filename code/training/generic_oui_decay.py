from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import torch
from torch import nn

from oui.core import compute_new_oui_from_activations
from training.llama_oui_decay import OUIDecayScheduler

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except Exception:  # pragma: no cover - optional depending on transformers version.
    HFConv1D = None


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _eligible_module_types() -> tuple[type[nn.Module], ...]:
    base_types: list[type[nn.Module]] = [nn.Conv2d, nn.Linear]
    if HFConv1D is not None:
        base_types.append(HFConv1D)
    return tuple(base_types)


def _is_decay_parameter(full_name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad:
        return False
    if full_name.endswith(".bias"):
        return False
    if param.ndim <= 1:
        return False
    return True


@dataclass
class GenericOptimizerGroupInfo:
    module_group_indices: Dict[str, int]
    other_decay_group_index: int | None
    no_decay_group_index: int | None


def list_dynamic_modules(
    model: nn.Module,
    *,
    remove_first_layer: bool = False,
    remove_last_layer: bool = False,
) -> "OrderedDict[str, nn.Module]":
    mapping: "OrderedDict[str, nn.Module]" = OrderedDict()
    eligible_types = _eligible_module_types()
    for name, module in _unwrap_model(model).named_modules():
        if not name:
            continue
        if not isinstance(module, eligible_types):
            continue
        weight = getattr(module, "weight", None)
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
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported AdamW fused mode: {adamw_fused_mode}")

    fused_supported = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if mode == "on":
        if not fused_supported:
            raise RuntimeError("AdamW fused mode requested, but this torch build has no 'fused' argument.")
        if device_type != "cuda":
            raise RuntimeError("AdamW fused mode requested, but device_type is not CUDA.")
        return {"fused": True}, True

    if mode == "off":
        if fused_supported:
            return {"fused": False}, False
        return {}, False

    if fused_supported and device_type == "cuda":
        return {"fused": True}, True
    if fused_supported:
        return {"fused": False}, False
    return {}, False


def build_standard_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    adamw_fused_mode: str = "auto",
    device_type: str = "cpu",
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
        param_groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas)
        setattr(optimizer, "_adamw_fused_effective", None)
        return optimizer
    if optimizer_name == "adamw":
        extra_kwargs, fused_effective = _resolve_adamw_fused_kwargs(
            adamw_fused_mode=adamw_fused_mode,
            device_type=device_type,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra_kwargs)
        setattr(optimizer, "_adamw_fused_effective", bool(fused_effective))
        return optimizer
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def list_decay_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    decay_params: list[torch.nn.Parameter] = []
    for full_name, param in _unwrap_model(model).named_parameters():
        if _is_decay_parameter(full_name, param):
            decay_params.append(param)
    return decay_params


def build_optimizer_with_module_groups(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    base_wd: float,
    dynamic_module_names: Sequence[str],
    betas: tuple[float, float],
    adamw_fused_mode: str = "auto",
    device_type: str = "cpu",
) -> tuple[torch.optim.Optimizer, GenericOptimizerGroupInfo]:
    module_map = list_dynamic_modules(model)
    missing = [name for name in dynamic_module_names if name not in module_map]
    if missing:
        raise ValueError(f"Unknown dynamic module names: {missing}")

    param_groups: List[MutableMapping[str, object]] = []
    assigned_ids = set()
    module_group_indices: Dict[str, int] = {}

    for module_name in dynamic_module_names:
        module = module_map[module_name]
        params: list[torch.nn.Parameter] = []
        for local_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{local_name}"
            if id(param) in assigned_ids or not _is_decay_parameter(full_name, param):
                continue
            params.append(param)
            assigned_ids.add(id(param))
        if not params:
            continue
        module_group_indices[module_name] = len(param_groups)
        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": float(base_wd),
                "group_name": module_name,
                "dynamic_wd": True,
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
                "params": other_decay_params,
                "lr": lr,
                "weight_decay": float(base_wd),
                "group_name": "other_decay",
                "dynamic_wd": False,
            }
        )

    no_decay_group_index = None
    if no_decay_params:
        no_decay_group_index = len(param_groups)
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
                "group_name": "no_decay",
                "dynamic_wd": False,
            }
        )

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas)
        setattr(optimizer, "_adamw_fused_effective", None)
    elif optimizer_name == "adamw":
        extra_kwargs, fused_effective = _resolve_adamw_fused_kwargs(
            adamw_fused_mode=adamw_fused_mode,
            device_type=device_type,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra_kwargs)
        setattr(optimizer, "_adamw_fused_effective", bool(fused_effective))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer, GenericOptimizerGroupInfo(
        module_group_indices=module_group_indices,
        other_decay_group_index=other_decay_group_index,
        no_decay_group_index=no_decay_group_index,
    )


class GenericOUICollector:
    def __init__(
        self,
        module_map: Mapping[str, nn.Module],
        *,
        sample_mode: str = "random",
        seed: int = 0,
    ) -> None:
        if sample_mode not in {"random", "first"}:
            raise ValueError("sample_mode must be 'random' or 'first'.")
        self.sample_mode = sample_mode
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._step_values: Dict[str, float] = {}

        for module_name, module in module_map.items():
            self._handles.append(module.register_forward_hook(self._make_hook(module_name)))

    def _select_indices(self, batch_size: int, num_positions: int, device: torch.device) -> torch.Tensor:
        if self.sample_mode == "first":
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if device.type == "cuda":
            return torch.randint(low=0, high=num_positions, size=(batch_size,), device=device)
        return torch.randint(
            low=0,
            high=num_positions,
            size=(batch_size,),
            generator=self.rng,
            device=device,
        )

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

    def step_values(self) -> Dict[str, float]:
        return dict(self._step_values)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _AlphaDecayBackboneNamespace(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone


class AlphaDecayCompatWrapper(nn.Module):
    """
    Give vendor AlphaDecay module names with at least three path components.

    Example:
    - original: conv1
    - wrapped: model.backbone.conv1
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.model = _AlphaDecayBackboneNamespace(backbone)

    def forward(self, *args, **kwargs):
        return self.model.backbone(*args, **kwargs)


def current_dynamic_wd(
    optimizer: torch.optim.Optimizer,
    module_group_indices: Mapping[str, int],
) -> Dict[str, float]:
    return {
        name: float(optimizer.param_groups[group_idx]["weight_decay"])
        for name, group_idx in module_group_indices.items()
    }
