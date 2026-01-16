"""
Retrieval stability metrics for curv-embedding.

Measures consistency of retrieval results across query reformulations
and against planted/expected results.

Metrics:
- hit_rate@k: Fraction of expected results retrieved in top-k
- reformulation_stability: Consistency of results across query rewrites
- topk_overlap_across_rewrites: Agreement between different query formulations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_hit_rate(
    retrieved: list[str],
    expected: set[str],
    k: int,
) -> float:
    """
    Compute hit rate at k against expected/planted results.

    Hit rate is the fraction of expected items that appear in the top-k
    retrieved results. This measures recall against known-relevant items.

    Args:
        retrieved: Ordered list of retrieved item IDs (by rank).
        expected: Set of expected/planted item IDs that should be retrieved.
        k: Number of top results to consider.

    Returns:
        Hit rate in range [0, 1]. Returns 0 if expected set is empty.

    Raises:
        ValueError: If k is non-positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not expected:
        return 0.0

    # Take top-k retrieved items
    top_k_retrieved = set(retrieved[:k])

    # Count hits
    hits = len(top_k_retrieved & expected)

    return hits / len(expected)


def compute_reformulation_stability(
    results_by_query: dict[str, list[str]],
    k: int,
) -> float:
    """
    Compute stability of results across query reformulations.

    Measures how consistent retrieval results are when the same intent
    is expressed through different query formulations. Computes mean
    pairwise overlap between all query pairs in a reformulation family.

    Args:
        results_by_query: Dict mapping query variant ID -> retrieved result IDs.
            All queries should be reformulations targeting the same intent.
        k: Number of top results to consider for overlap.

    Returns:
        Mean pairwise overlap in range [0, 1].
        Returns 1.0 if only one query (trivially stable).
        Returns 0.0 if no queries provided.

    Raises:
        ValueError: If k is non-positive or any result list is shorter than k.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not results_by_query:
        return 0.0

    if len(results_by_query) == 1:
        return 1.0

    # Get sorted keys for deterministic ordering
    query_ids = sorted(results_by_query.keys())

    # Validate all lists have at least k items
    for query_id in query_ids:
        if len(results_by_query[query_id]) < k:
            raise ValueError(
                f"Query {query_id} has {len(results_by_query[query_id])} results, "
                f"need at least {k}"
            )

    # Compute pairwise overlaps
    overlaps: list[float] = []
    for q1, q2 in combinations(query_ids, 2):
        results1 = set(results_by_query[q1][:k])
        results2 = set(results_by_query[q2][:k])
        overlap = len(results1 & results2) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


@dataclass
class FamilyStats:
    """
    Statistics for a single query family.

    A query family is a set of query reformulations targeting the same intent.

    Attributes:
        family_id: Identifier for the query family.
        mean_hit_rate: Mean hit rate across k values and queries.
        mean_stability: Mean pairwise stability within family.
        hit_rates_by_k: Dict mapping k -> mean hit rate at that k.
        num_queries: Number of query variants in family.
    """

    family_id: str
    mean_hit_rate: float
    mean_stability: float
    hit_rates_by_k: dict[int, float]
    num_queries: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "family_id": self.family_id,
            "mean_hit_rate": self.mean_hit_rate,
            "mean_stability": self.mean_stability,
            "hit_rates_by_k": self.hit_rates_by_k,
            "num_queries": self.num_queries,
        }


@dataclass
class OverlapResult:
    """
    Aggregated retrieval stability statistics.

    Attributes:
        mean_hit_rate: Mean hit rate across all families and k values.
        mean_stability: Mean reformulation stability across families.
        hit_rates_by_k: Dict mapping k -> overall mean hit rate at that k.
        per_family_stats: List of per-family statistics.
        num_families: Number of query families evaluated.
        k_values: List of k values used in evaluation.
    """

    mean_hit_rate: float
    mean_stability: float
    hit_rates_by_k: dict[int, float]
    per_family_stats: list[FamilyStats] = field(repr=False)
    num_families: int
    k_values: list[int]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_hit_rate": self.mean_hit_rate,
            "mean_stability": self.mean_stability,
            "hit_rates_by_k": self.hit_rates_by_k,
            "num_families": self.num_families,
            "k_values": self.k_values,
            "per_family_stats": [fs.to_dict() for fs in self.per_family_stats],
        }


def compute_overlap_stats(
    query_families: dict[str, list[str]],
    results: dict[str, list[str]],
    expected: dict[str, set[str]],
    k_values: list[int],
) -> OverlapResult:
    """
    Compute comprehensive retrieval stability statistics.

    Evaluates hit rates and reformulation stability across query families,
    where each family contains multiple query reformulations targeting
    the same known-relevant results.

    Args:
        query_families: Dict mapping family_id -> list of query IDs in that family.
        results: Dict mapping query_id -> list of retrieved item IDs (ordered by rank).
        expected: Dict mapping family_id -> set of expected/planted item IDs.
        k_values: List of k values to evaluate hit rate at.

    Returns:
        OverlapResult with comprehensive statistics.

    Raises:
        ValueError: If inputs are empty or inconsistent.
    """
    if not query_families:
        raise ValueError("query_families cannot be empty")
    if not k_values:
        raise ValueError("k_values cannot be empty")
    if not all(k > 0 for k in k_values):
        raise ValueError("All k values must be positive")

    per_family_stats: list[FamilyStats] = []
    all_hit_rates_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    all_stabilities: list[float] = []

    # Sort families for determinism
    for family_id in sorted(query_families.keys()):
        query_ids = query_families[family_id]
        family_expected = expected.get(family_id, set())

        # Gather results for this family's queries
        family_results: dict[str, list[str]] = {}
        for query_id in query_ids:
            if query_id in results:
                family_results[query_id] = results[query_id]

        if not family_results:
            # Skip family if no results available
            continue

        # Compute hit rates at each k
        family_hit_rates_by_k: dict[int, float] = {}
        for k in k_values:
            hit_rates = [
                compute_hit_rate(family_results[q], family_expected, k)
                for q in sorted(family_results.keys())
            ]
            mean_hr = float(np.mean(hit_rates)) if hit_rates else 0.0
            family_hit_rates_by_k[k] = mean_hr
            all_hit_rates_by_k[k].append(mean_hr)

        # Compute stability within family
        # Need to find max k that all results support
        min_result_len = min(len(r) for r in family_results.values())
        stability_k = min(max(k_values), min_result_len)
        if stability_k > 0:
            stability = compute_reformulation_stability(family_results, stability_k)
        else:
            stability = 0.0
        all_stabilities.append(stability)

        per_family_stats.append(
            FamilyStats(
                family_id=family_id,
                mean_hit_rate=float(np.mean(list(family_hit_rates_by_k.values()))),
                mean_stability=stability,
                hit_rates_by_k=family_hit_rates_by_k,
                num_queries=len(family_results),
            )
        )

    # Aggregate across families
    overall_hit_rates_by_k = {
        k: float(np.mean(rates)) if rates else 0.0
        for k, rates in all_hit_rates_by_k.items()
    }

    overall_hit_rate = (
        float(np.mean(list(overall_hit_rates_by_k.values())))
        if overall_hit_rates_by_k
        else 0.0
    )

    overall_stability = float(np.mean(all_stabilities)) if all_stabilities else 0.0

    return OverlapResult(
        mean_hit_rate=overall_hit_rate,
        mean_stability=overall_stability,
        hit_rates_by_k=overall_hit_rates_by_k,
        per_family_stats=per_family_stats,
        num_families=len(per_family_stats),
        k_values=k_values,
    )
