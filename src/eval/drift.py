"""
Embedding drift metrics for curv-embedding.

Measures how embeddings change across model revisions for identical content.
Chunks are matched by content SHA256 hash to ensure we compare identical text.

Metrics:
- drift_cos: 1 - cosine_similarity(e_old, e_new)
- drift_l2: L2 norm of the difference ||e_old - e_new||_2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_drift_cosine(e_old: NDArray[np.floating], e_new: NDArray[np.floating]) -> float:
    """
    Compute cosine drift between two embedding vectors.

    Drift is defined as 1 - cosine_similarity, so:
    - 0.0 means identical direction (no drift)
    - 1.0 means orthogonal vectors
    - 2.0 means opposite direction (maximum drift)

    Args:
        e_old: Old embedding vector (1D array).
        e_new: New embedding vector (1D array), same dimension as e_old.

    Returns:
        Cosine drift value in range [0, 2].

    Raises:
        ValueError: If vectors have different shapes or zero norm.
    """
    e_old = np.asarray(e_old, dtype=np.float64)
    e_new = np.asarray(e_new, dtype=np.float64)

    if e_old.shape != e_new.shape:
        raise ValueError(
            f"Embedding shapes must match: {e_old.shape} vs {e_new.shape}"
        )

    if e_old.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got shape {e_old.shape}")

    norm_old = np.linalg.norm(e_old)
    norm_new = np.linalg.norm(e_new)

    if norm_old == 0 or norm_new == 0:
        raise ValueError("Cannot compute cosine similarity for zero-norm vectors")

    cosine_sim = np.dot(e_old, e_new) / (norm_old * norm_new)
    # Clip to handle floating point errors that could put us slightly outside [-1, 1]
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    return float(1.0 - cosine_sim)


def compute_drift_l2(e_old: NDArray[np.floating], e_new: NDArray[np.floating]) -> float:
    """
    Compute L2 (Euclidean) drift between two embedding vectors.

    Args:
        e_old: Old embedding vector (1D array).
        e_new: New embedding vector (1D array), same dimension as e_old.

    Returns:
        L2 distance between the vectors (non-negative).

    Raises:
        ValueError: If vectors have different shapes.
    """
    e_old = np.asarray(e_old, dtype=np.float64)
    e_new = np.asarray(e_new, dtype=np.float64)

    if e_old.shape != e_new.shape:
        raise ValueError(
            f"Embedding shapes must match: {e_old.shape} vs {e_new.shape}"
        )

    if e_old.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got shape {e_old.shape}")

    return float(np.linalg.norm(e_old - e_new))


@dataclass
class DriftResult:
    """
    Aggregated drift statistics across matched embedding pairs.

    Attributes:
        mean_cosine: Mean cosine drift across all pairs.
        p90_cosine: 90th percentile of cosine drift.
        mean_l2: Mean L2 drift across all pairs.
        p90_l2: 90th percentile of L2 drift.
        cosine_distribution: Array of all cosine drift values.
        l2_distribution: Array of all L2 drift values.
        num_matched: Number of embedding pairs matched by content hash.
        num_unmatched_old: Embeddings in old set not found in new set.
        num_unmatched_new: Embeddings in new set not found in old set.
    """

    mean_cosine: float
    p90_cosine: float
    mean_l2: float
    p90_l2: float
    cosine_distribution: NDArray[np.floating] = field(repr=False)
    l2_distribution: NDArray[np.floating] = field(repr=False)
    num_matched: int
    num_unmatched_old: int
    num_unmatched_new: int

    def __post_init__(self) -> None:
        """Ensure distributions are numpy arrays."""
        self.cosine_distribution = np.asarray(self.cosine_distribution, dtype=np.float64)
        self.l2_distribution = np.asarray(self.l2_distribution, dtype=np.float64)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_cosine": self.mean_cosine,
            "p90_cosine": self.p90_cosine,
            "mean_l2": self.mean_l2,
            "p90_l2": self.p90_l2,
            "num_matched": self.num_matched,
            "num_unmatched_old": self.num_unmatched_old,
            "num_unmatched_new": self.num_unmatched_new,
            "cosine_distribution": self.cosine_distribution.tolist(),
            "l2_distribution": self.l2_distribution.tolist(),
        }


def compute_drift_stats(
    old_embeddings: dict[str, NDArray[np.floating]],
    new_embeddings: dict[str, NDArray[np.floating]],
) -> DriftResult:
    """
    Compute drift statistics for embeddings matched by content SHA256.

    Compares embeddings of identical chunks (matched by their content hash key)
    across two embedding sets, typically from different model versions.

    Args:
        old_embeddings: Dict mapping content_sha256 -> embedding vector (old model).
        new_embeddings: Dict mapping content_sha256 -> embedding vector (new model).

    Returns:
        DriftResult with aggregated statistics.

    Raises:
        ValueError: If no matching content hashes are found.
    """
    # Find matching content hashes
    old_keys = set(old_embeddings.keys())
    new_keys = set(new_embeddings.keys())
    matched_keys = old_keys & new_keys

    if not matched_keys:
        raise ValueError("No matching content hashes found between old and new embeddings")

    # Compute drift for each matched pair
    cosine_drifts: list[float] = []
    l2_drifts: list[float] = []

    for key in sorted(matched_keys):  # Sort for determinism
        e_old = old_embeddings[key]
        e_new = new_embeddings[key]

        cosine_drifts.append(compute_drift_cosine(e_old, e_new))
        l2_drifts.append(compute_drift_l2(e_old, e_new))

    cosine_arr = np.array(cosine_drifts, dtype=np.float64)
    l2_arr = np.array(l2_drifts, dtype=np.float64)

    return DriftResult(
        mean_cosine=float(np.mean(cosine_arr)),
        p90_cosine=float(np.percentile(cosine_arr, 90)),
        mean_l2=float(np.mean(l2_arr)),
        p90_l2=float(np.percentile(l2_arr, 90)),
        cosine_distribution=cosine_arr,
        l2_distribution=l2_arr,
        num_matched=len(matched_keys),
        num_unmatched_old=len(old_keys - new_keys),
        num_unmatched_new=len(new_keys - old_keys),
    )
