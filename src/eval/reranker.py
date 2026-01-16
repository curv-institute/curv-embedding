"""
Representational reranker for curv-embedding v2.0.0.

Implements reranking over ANN candidates using stored stability
and strain diagnostics from SQLite metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import RerankConfig
    from src.storage.sqlite_store import ChunkRecord


@dataclass
class RerankResult:
    """Result from reranking operation."""

    chunk_ids: list[str]
    scores: list[float]
    original_ranks: list[int]
    rerank_mode: str


class RepresentationalReranker:
    """
    Reranks ANN candidates using representational diagnostics.

    v2.0.0: Uses stored stability and strain signals to improve
    retrieval consistency beyond pure cosine similarity.
    """

    def __init__(self, config: RerankConfig) -> None:
        """
        Initialize the reranker.

        Args:
            config: RerankConfig with weights and parameters.
        """
        self._config = config
        self._rng = random.Random(config.seed)

        # Precompute normalization factors for signals
        self._signal_stats: dict[str, tuple[float, float]] = {}

    def compute_strain(self, chunk: ChunkRecord) -> float:
        """
        Compute strain score from stored diagnostics.

        Strain represents how far the chunk is from stable representation.
        Higher strain = less stable embedding.

        Args:
            chunk: ChunkRecord with diagnostic signals.

        Returns:
            Normalized strain score in [0, 1].
        """
        # Use curvature and stability margin as strain indicators
        # High curvature + low stability margin = high strain
        curvature = chunk.curvature_signal or 0.0
        stability = chunk.stability_margin_signal or 1.0

        # Normalize curvature (higher is worse)
        # Typical range is 0-1 based on percentile scoring
        curvature_norm = min(1.0, max(0.0, curvature))

        # Invert stability (low stability = high strain)
        # Stability margin is typically 0-1
        stability_penalty = 1.0 - min(1.0, max(0.0, stability))

        # Combine with equal weights
        strain = 0.5 * curvature_norm + 0.5 * stability_penalty

        return strain

    def compute_boundary_penalty(self, chunk: ChunkRecord) -> float:
        """
        Compute boundary penalty from chunk metadata.

        Penalizes chunks near unstable or edit-prone boundaries.

        Args:
            chunk: ChunkRecord with boundary information.

        Returns:
            Penalty score in [0, 1].
        """
        penalty = 0.0

        # Structural boundaries are natural break points (lower penalty)
        # Non-structural boundaries may be arbitrary (higher penalty)
        if not chunk.is_structural_boundary:
            penalty += 0.3

        # Low cut score indicates weak boundary choice
        cut_score = chunk.cut_score or 0.5
        if cut_score < 0.3:
            penalty += 0.4
        elif cut_score < 0.5:
            penalty += 0.2

        # Check for disharmony if available
        disharmony = chunk.disharmony_signal or 0.0
        if disharmony > 0.5:
            penalty += 0.3

        return min(1.0, penalty)

    def compute_instability_risk(
        self,
        chunk: ChunkRecord,
        chunk_size_bytes: int | None = None,
    ) -> float:
        """
        Compute instability risk based on historical patterns.

        Uses chunk size as proxy for mutation cost (larger = more risk).

        Args:
            chunk: ChunkRecord with size information.
            chunk_size_bytes: Optional precomputed chunk size.

        Returns:
            Risk score in [0, 1].
        """
        # Use chunk size as proxy for mutation cost
        # Larger chunks are more vulnerable to edits
        if chunk_size_bytes is None:
            chunk_size_bytes = chunk.byte_offset_end - chunk.byte_offset_start

        # Normalize based on typical chunk sizes (256-4096 bytes)
        size_risk = min(1.0, chunk_size_bytes / 4096)

        return size_risk

    def rerank(
        self,
        candidates: list[tuple[str, float, ChunkRecord]],
        mode: str | None = None,
    ) -> RerankResult:
        """
        Rerank candidates using the configured mode.

        Args:
            candidates: List of (chunk_id, cosine_similarity, ChunkRecord) tuples.
            mode: Override mode ("ann", "ann_random", "ann_repr").
                  Uses config.mode if not specified.

        Returns:
            RerankResult with reranked chunk IDs and scores.
        """
        if not candidates:
            return RerankResult(
                chunk_ids=[],
                scores=[],
                original_ranks=[],
                rerank_mode=mode or self._config.mode,
            )

        mode = mode or self._config.mode
        k = self._config.k

        if mode == "ann":
            # Pure ANN - no reranking, just take top-k
            return self._ann_only(candidates, k)

        elif mode == "ann_random":
            # Random rerank control
            return self._ann_random(candidates, k)

        elif mode == "ann_repr":
            # Representational reranking
            return self._ann_repr(candidates, k)

        else:
            raise ValueError(f"Unknown rerank mode: {mode}")

    def _ann_only(
        self,
        candidates: list[tuple[str, float, ChunkRecord]],
        k: int,
    ) -> RerankResult:
        """Return top-k by cosine similarity only."""
        # Candidates should already be sorted by similarity
        top_k = candidates[:k]

        return RerankResult(
            chunk_ids=[c[0] for c in top_k],
            scores=[c[1] for c in top_k],
            original_ranks=list(range(len(top_k))),
            rerank_mode="ann",
        )

    def _ann_random(
        self,
        candidates: list[tuple[str, float, ChunkRecord]],
        k: int,
    ) -> RerankResult:
        """Random rerank control - shuffle candidates then take top-k."""
        # Record original ranks
        original_ranks = {c[0]: i for i, c in enumerate(candidates)}

        # Shuffle candidates
        shuffled = list(candidates)
        self._rng.shuffle(shuffled)

        # Take top-k from shuffled list
        top_k = shuffled[:k]

        return RerankResult(
            chunk_ids=[c[0] for c in top_k],
            scores=[c[1] for c in top_k],
            original_ranks=[original_ranks[c[0]] for c in top_k],
            rerank_mode="ann_random",
        )

    def _ann_repr(
        self,
        candidates: list[tuple[str, float, ChunkRecord]],
        k: int,
    ) -> RerankResult:
        """
        Representational reranking.

        score(c) = alpha * sim_cos(c)
                 - beta  * strain(c)
                 - gamma * boundary_penalty(c)
                 - delta * instability_risk(c)
        """
        alpha = self._config.alpha
        beta = self._config.beta
        gamma = self._config.gamma
        delta = self._config.delta

        # Record original ranks
        original_ranks = {c[0]: i for i, c in enumerate(candidates)}

        # Compute rerank scores
        scored: list[tuple[str, float, float, int]] = []

        for chunk_id, sim_cos, chunk in candidates:
            strain = self.compute_strain(chunk)
            boundary = self.compute_boundary_penalty(chunk)
            risk = self.compute_instability_risk(chunk)

            score = (
                alpha * sim_cos
                - beta * strain
                - gamma * boundary
                - delta * risk
            )

            scored.append((chunk_id, score, sim_cos, original_ranks[chunk_id]))

        # Sort by rerank score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top_k = scored[:k]

        return RerankResult(
            chunk_ids=[c[0] for c in top_k],
            scores=[c[1] for c in top_k],
            original_ranks=[c[3] for c in top_k],
            rerank_mode="ann_repr",
        )

    def compute_disagreement(
        self,
        ann_result: RerankResult,
        repr_result: RerankResult,
    ) -> float:
        """
        Compute disagreement rate between ANN and repr reranking.

        Args:
            ann_result: Result from ann mode.
            repr_result: Result from ann_repr mode.

        Returns:
            Fraction of positions where the two rankings differ.
        """
        if not ann_result.chunk_ids or not repr_result.chunk_ids:
            return 0.0

        k = len(ann_result.chunk_ids)
        matches = sum(
            1 for i in range(k)
            if i < len(repr_result.chunk_ids)
            and ann_result.chunk_ids[i] == repr_result.chunk_ids[i]
        )

        return 1.0 - (matches / k)
