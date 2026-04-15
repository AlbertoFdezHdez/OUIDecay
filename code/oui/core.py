from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OUIComputationStats:
    batch_size: int
    feature_dim: int
    half_batch: int
    activation_shape: tuple[int, ...]
    binary_matrix_bytes: int
    positives_bytes: int
    minority_bytes: int
    estimated_working_set_bytes: int


def flatten_activations_per_sample(activations: torch.Tensor) -> torch.Tensor:
    """Return activations as a B x d matrix."""
    if activations.ndim < 2:
        raise ValueError(
            f"Expected activations with at least 2 dims (B x ...), got shape {tuple(activations.shape)}"
        )
    return activations.reshape(activations.shape[0], -1)


@torch.no_grad()
def compute_new_oui_from_activations(activations: torch.Tensor) -> float:
    oui, _ = compute_new_oui_with_stats(activations)
    return oui


@torch.no_grad()
def compute_new_oui_with_stats(activations: torch.Tensor) -> tuple[float, OUIComputationStats]:
    """
    Compute batch-based OUI for a module from activations.

    OUI_l = (1 / d_l) * sum_j [ min(s_j, B - s_j) / floor(B/2) ]
    where s_j is the number of positives in feature j.
    """
    matrix = flatten_activations_per_sample(activations)
    batch_size = matrix.shape[0]
    feature_dim = matrix.shape[1]
    half_batch = batch_size // 2
    activation_shape = tuple(activations.shape)

    binary_matrix_bytes = matrix.numel()  # bool tensor expected from (matrix > 0)
    positives_bytes = feature_dim * 8  # torch.sum over bool yields int64 by default
    minority_bytes = feature_dim * 4  # converted to float32
    estimated_working_set_bytes = binary_matrix_bytes + positives_bytes + minority_bytes

    stats = OUIComputationStats(
        batch_size=batch_size,
        feature_dim=feature_dim,
        half_batch=half_batch,
        activation_shape=activation_shape,
        binary_matrix_bytes=binary_matrix_bytes,
        positives_bytes=positives_bytes,
        minority_bytes=minority_bytes,
        estimated_working_set_bytes=estimated_working_set_bytes,
    )

    if batch_size < 2 or half_batch == 0:
        return 0.0, stats

    positives = (matrix > 0).sum(dim=0)
    minority = torch.minimum(positives, batch_size - positives).to(torch.float32)
    oui = (minority / float(half_batch)).mean()
    return float(oui.item()), stats


def select_token_indices(
    batch_size: int,
    num_tokens: int,
    mode: str,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select one token index per sample."""
    if mode == "cls":
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if mode == "random":
        return torch.randint(
            low=0,
            high=num_tokens,
            size=(batch_size,),
            device=device,
            generator=generator,
        )
    raise ValueError(f"Unsupported token selection mode: {mode}")


def gather_token_per_sample(features: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """
    Gather one token per sample.

    Supports:
    - B x T x C -> B x C
    - B x H x T x D -> B x H x D
    """
    if features.ndim == 3:
        bsz, _, channels = features.shape
        gather_index = token_indices.view(bsz, 1, 1).expand(-1, 1, channels)
        return features.gather(dim=1, index=gather_index).squeeze(1)

    if features.ndim == 4:
        bsz, heads, _, head_dim = features.shape
        gather_index = token_indices.view(bsz, 1, 1, 1).expand(-1, heads, 1, head_dim)
        return features.gather(dim=2, index=gather_index).squeeze(2)

    raise ValueError(
        f"Unsupported tensor shape for per-sample token gather: {tuple(features.shape)}"
    )
