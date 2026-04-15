from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import torch

from .core import OUIComputationStats, compute_new_oui_with_stats


@dataclass
class OUIRecord:
    name: str
    depth: int
    module_type: str
    sum_oui: float = 0.0
    count: int = 0
    last_oui: float = 0.0
    total_samples: int = 0
    last_batch_size: int = 0
    last_feature_dim: int = 0
    last_half_batch: int = 0
    last_activation_shape: tuple[int, ...] = ()
    last_working_set_bytes: int = 0
    peak_working_set_bytes: int = 0

    @property
    def mean_oui(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_oui / self.count


class OUITracker:
    """Online OUI tracker computed during forward passes."""

    def __init__(self) -> None:
        self._records: "OrderedDict[str, OUIRecord]" = OrderedDict()
        self._last_step: Dict[str, float] = {}

    def reset_step(self) -> None:
        self._last_step = {}

    @torch.no_grad()
    def track(
        self,
        name: str,
        depth: int,
        activations: torch.Tensor,
        module_type: str,
    ) -> float:
        oui_value, stats = compute_new_oui_with_stats(activations)

        if name not in self._records:
            self._records[name] = OUIRecord(name=name, depth=depth, module_type=module_type)

        record = self._records[name]
        record.last_oui = oui_value
        record.sum_oui += oui_value
        record.count += 1
        self._update_stats(record, stats)
        self._last_step[name] = oui_value
        return oui_value

    @staticmethod
    def _update_stats(record: OUIRecord, stats: OUIComputationStats) -> None:
        record.total_samples += stats.batch_size
        record.last_batch_size = stats.batch_size
        record.last_feature_dim = stats.feature_dim
        record.last_half_batch = stats.half_batch
        record.last_activation_shape = stats.activation_shape
        record.last_working_set_bytes = stats.estimated_working_set_bytes
        record.peak_working_set_bytes = max(
            record.peak_working_set_bytes, stats.estimated_working_set_bytes
        )

    def last_step_metrics(self) -> Dict[str, float]:
        return dict(self._last_step)

    def mean_metrics(self) -> Dict[str, float]:
        return {name: record.mean_oui for name, record in self._records.items()}

    def rows(self) -> List[Dict[str, object]]:
        ordered = sorted(self._records.values(), key=lambda x: (x.depth, x.name))
        return [
            {
                "name": record.name,
                "depth": record.depth,
                "module_type": record.module_type,
                "mean_oui": record.mean_oui,
                "last_oui": record.last_oui,
                "count": record.count,
                "total_samples": record.total_samples,
                "last_batch_size": record.last_batch_size,
                "last_feature_dim": record.last_feature_dim,
                "last_half_batch": record.last_half_batch,
                "last_activation_shape": record.last_activation_shape,
                "last_working_set_bytes": record.last_working_set_bytes,
                "peak_working_set_bytes": record.peak_working_set_bytes,
            }
            for record in ordered
        ]

    def total_oui_calls(self) -> int:
        return sum(record.count for record in self._records.values())

    def total_samples_used(self) -> int:
        return sum(record.total_samples for record in self._records.values())

    def peak_working_set_bytes(self) -> int:
        if not self._records:
            return 0
        return max(record.peak_working_set_bytes for record in self._records.values())
