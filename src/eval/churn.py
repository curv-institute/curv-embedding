"""
ANN neighbor churn metrics for curv-embedding.

Measures how nearest neighbor results change across model/index revisions
for a fixed probe set. This captures the practical impact of embedding drift
on retrieval results.

Metrics:
- topk_overlap: |N_k^old intersection N_k^new| / k
- jaccard_k: |intersection| / |union| for top-k neighbors
- rank_correlation: Kendall tau on ranks for shared neighbors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_topk_overlap(
    old_neighbors: list[str],
    new_neighbors: list[str],
    k: int,
) -> float:
    """
    Compute top-k overlap between two neighbor lists.

    Overlap is defined as the size of the intersection divided by k,
    measuring what fraction of the top-k results are preserved.

    Args:
        old_neighbors: List of neighbor IDs from old model/index (ordered by rank).
        new_neighbors: List of neighbor IDs from new model/index (ordered by rank).
        k: Number of top neighbors to consider.

    Returns:
        Overlap ratio in range [0, 1].

    Raises:
        ValueError: If k is non-positive or lists are shorter than k.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if len(old_neighbors) < k:
        raise ValueError(
            f"old_neighbors has {len(old_neighbors)} items, need at least {k}"
        )
    if len(new_neighbors) < k:
        raise ValueError(
            f"new_neighbors has {len(new_neighbors)} items, need at least {k}"
        )

    old_set = set(old_neighbors[:k])
    new_set = set(new_neighbors[:k])

    intersection_size = len(old_set & new_set)
    return intersection_size / k


def compute_jaccard(
    old_neighbors: list[str],
    new_neighbors: list[str],
) -> float:
    """
    Compute Jaccard similarity between two neighbor lists.

    Jaccard index is |intersection| / |union|, providing a set-based
    similarity measure that accounts for both preserved and changed results.

    Args:
        old_neighbors: List of neighbor IDs from old model/index.
        new_neighbors: List of neighbor IDs from new model/index.

    Returns:
        Jaccard index in range [0, 1]. Returns 0 if both lists are empty.
    """
    old_set = set(old_neighbors)
    new_set = set(new_neighbors)

    if not old_set and not new_set:
        return 0.0

    intersection_size = len(old_set & new_set)
    union_size = len(old_set | new_set)

    return intersection_size / union_size


def compute_rank_correlation(
    old_neighbors: list[str],
    new_neighbors: list[str],
) -> float:
    """
    Compute Kendall tau rank correlation for shared neighbors.

    For neighbors appearing in both lists, computes Kendall's tau-b
    correlation between their ranks. This measures whether the relative
    ordering of results is preserved even when the sets differ.

    Args:
        old_neighbors: Ordered list of neighbor IDs from old model/index.
        new_neighbors: Ordered list of neighbor IDs from new model/index.

    Returns:
        Kendall tau correlation in range [-1, 1].
        Returns 0.0 if fewer than 2 shared neighbors (correlation undefined).
    """
    # Build rank mappings (1-indexed for interpretability)
    old_ranks = {neighbor: rank + 1 for rank, neighbor in enumerate(old_neighbors)}
    new_ranks = {neighbor: rank + 1 for rank, neighbor in enumerate(new_neighbors)}

    # Find shared neighbors
    shared = set(old_ranks.keys()) & set(new_ranks.keys())

    if len(shared) < 2:
        # Correlation undefined with fewer than 2 points
        return 0.0

    # Extract ranks for shared neighbors in deterministic order
    shared_sorted = sorted(shared)
    old_rank_values = [old_ranks[n] for n in shared_sorted]
    new_rank_values = [new_ranks[n] for n in shared_sorted]

    # Compute Kendall tau-b (handles ties correctly)
    tau, _ = stats.kendalltau(old_rank_values, new_rank_values)

    # Handle NaN case (can occur with constant ranks)
    if np.isnan(tau):
        return 0.0

    return float(tau)


@dataclass
class ChurnResult:
    """
    Aggregated churn statistics across probe queries.

    Attributes:
        mean_overlap: Mean top-k overlap across all probe queries.
        p10_overlap: 10th percentile of overlap (worst-case indicator).
        mean_jaccard: Mean Jaccard similarity across all queries.
        mean_rank_correlation: Mean Kendall tau for shared neighbors.
        overlap_distribution: Array of all overlap values.
        jaccard_distribution: Array of all Jaccard values.
        rank_correlation_distribution: Array of all rank correlations.
        k: The k value used for top-k calculations.
        num_queries: Number of probe queries evaluated.
    """

    mean_overlap: float
    p10_overlap: float
    mean_jaccard: float
    mean_rank_correlation: float
    overlap_distribution: NDArray[np.floating] = field(repr=False)
    jaccard_distribution: NDArray[np.floating] = field(repr=False)
    rank_correlation_distribution: NDArray[np.floating] = field(repr=False)
    k: int
    num_queries: int

    def __post_init__(self) -> None:
        """Ensure distributions are numpy arrays."""
        self.overlap_distribution = np.asarray(self.overlap_distribution, dtype=np.float64)
        self.jaccard_distribution = np.asarray(self.jaccard_distribution, dtype=np.float64)
        self.rank_correlation_distribution = np.asarray(
            self.rank_correlation_distribution, dtype=np.float64
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_overlap": self.mean_overlap,
            "p10_overlap": self.p10_overlap,
            "mean_jaccard": self.mean_jaccard,
            "mean_rank_correlation": self.mean_rank_correlation,
            "k": self.k,
            "num_queries": self.num_queries,
            "overlap_distribution": self.overlap_distribution.tolist(),
            "jaccard_distribution": self.jaccard_distribution.tolist(),
            "rank_correlation_distribution": self.rank_correlation_distribution.tolist(),
        }


def compute_churn_stats(
    old_neighbor_lists: list[list[str]],
    new_neighbor_lists: list[list[str]],
    k: int,
) -> ChurnResult:
    """
    Compute churn statistics for a set of probe queries.

    Compares nearest neighbor results between old and new embeddings/indices
    for a fixed set of probe queries.

    Args:
        old_neighbor_lists: List of neighbor ID lists from old model/index.
            Each inner list is ordered by rank (closest first).
        new_neighbor_lists: List of neighbor ID lists from new model/index.
            Must have same length as old_neighbor_lists.
        k: Number of top neighbors to consider for overlap calculations.

    Returns:
        ChurnResult with aggregated statistics.

    Raises:
        ValueError: If input lists have different lengths or are empty.
    """
    if len(old_neighbor_lists) != len(new_neighbor_lists):
        raise ValueError(
            f"List lengths must match: {len(old_neighbor_lists)} vs {len(new_neighbor_lists)}"
        )

    if not old_neighbor_lists:
        raise ValueError("Cannot compute churn stats for empty query lists")

    overlaps: list[float] = []
    jaccards: list[float] = []
    rank_correlations: list[float] = []

    for old_neighbors, new_neighbors in zip(
        old_neighbor_lists, new_neighbor_lists, strict=True
    ):
        overlaps.append(compute_topk_overlap(old_neighbors, new_neighbors, k))
        jaccards.append(compute_jaccard(old_neighbors[:k], new_neighbors[:k]))
        rank_correlations.append(compute_rank_correlation(old_neighbors, new_neighbors))

    overlap_arr = np.array(overlaps, dtype=np.float64)
    jaccard_arr = np.array(jaccards, dtype=np.float64)
    rank_corr_arr = np.array(rank_correlations, dtype=np.float64)

    return ChurnResult(
        mean_overlap=float(np.mean(overlap_arr)),
        p10_overlap=float(np.percentile(overlap_arr, 10)),
        mean_jaccard=float(np.mean(jaccard_arr)),
        mean_rank_correlation=float(np.mean(rank_corr_arr)),
        overlap_distribution=overlap_arr,
        jaccard_distribution=jaccard_arr,
        rank_correlation_distribution=rank_corr_arr,
        k=k,
        num_queries=len(old_neighbor_lists),
    )
