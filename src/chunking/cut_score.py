"""
Cut-score computation for stability-driven chunking.

This module provides the core cut-score algorithm that estimates representational
instability at each position in a document. Higher cut-scores indicate better
chunk boundary candidates.

The cut-score formula is:
    cut_score(t) =
      wK * relu(K~_t - k0)
    + wD * relu(D~_t - d0)
    + wS * relu(s0 - S~_t)
    + wB * B_t
    + wL * relu((L_t - L_target) / L_target)

Where:
    K_t: curvature signal (higher = more strain)
    S_t: stability margin (higher = more stable)
    D_t: disharmony signal (higher = less coherent)
    B_t: structural boundary indicator (0/1)
    L_t: current chunk length in bytes

Signals are normalized via z-score within a trailing window.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import ChunkingConfig


@dataclass
class CutScoreSignals:
    """Raw signals used for cut-score computation.

    Attributes:
        K: Curvature signal - higher values indicate more representational strain.
            In v0.1, simulated as local byte entropy.
        S: Stability margin - higher values indicate more stable representations.
            In v0.1, simulated as inverse of byte variance.
        D: Disharmony signal - higher values indicate less coherent representations.
            Disabled (0.0) in v0.1 without HHC.
        B: Structural boundary indicator - 1.0 at structural boundaries (newlines),
            0.0 elsewhere.
        L: Current chunk length in bytes from the last boundary.
    """

    K: float = 0.0
    S: float = 0.0
    D: float = 0.0
    B: float = 0.0
    L: int = 0


@dataclass
class NormalizedSignals:
    """Z-score normalized signals for cut-score computation.

    These are the tilde (~) versions of the raw signals, normalized
    within a trailing window.
    """

    K_norm: float = 0.0
    S_norm: float = 0.0
    D_norm: float = 0.0
    B: float = 0.0  # B is not normalized (binary indicator)
    L: int = 0  # L is used directly in length penalty


def relu(x: float) -> float:
    """Rectified Linear Unit activation function.

    Args:
        x: Input value.

    Returns:
        max(0, x) - returns x if positive, else 0.
    """
    return max(0.0, x)


@dataclass
class RollingNormalizer:
    """Streaming z-score normalization over a trailing window.

    Maintains running statistics for efficient incremental normalization
    of streaming signals.

    Attributes:
        window_size: Number of samples to maintain in the window.
        min_samples: Minimum samples required before normalization is applied.
            Below this threshold, raw values are returned.
    """

    window_size: int = 1024
    min_samples: int = 10
    _values: deque[float] = field(default_factory=deque)
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def __post_init__(self) -> None:
        """Initialize the deque with maxlen."""
        self._values = deque(maxlen=self.window_size)

    def update(self, value: float) -> float:
        """Add a new value and return its z-score.

        Args:
            value: The raw signal value to normalize.

        Returns:
            Z-score normalized value, or raw value if insufficient samples.
        """
        # Remove oldest value from running stats if window is full
        if len(self._values) == self.window_size:
            old_value = self._values[0]
            self._sum -= old_value
            self._sum_sq -= old_value * old_value

        # Add new value
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

        # Return raw value if insufficient samples
        n = len(self._values)
        if n < self.min_samples:
            return value

        # Compute z-score
        mean = self._sum / n
        variance = (self._sum_sq / n) - (mean * mean)

        # Handle numerical precision issues
        if variance < 1e-10:
            return 0.0

        std = math.sqrt(variance)
        return (value - mean) / std

    def reset(self) -> None:
        """Reset the normalizer state."""
        self._values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0

    @property
    def count(self) -> int:
        """Return the current number of samples in the window."""
        return len(self._values)


@dataclass
class SignalNormalizers:
    """Collection of normalizers for all signals.

    Each signal type has its own normalizer to maintain independent
    statistics.
    """

    window_size: int = 1024
    K_normalizer: RollingNormalizer = field(init=False)
    S_normalizer: RollingNormalizer = field(init=False)
    D_normalizer: RollingNormalizer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize individual normalizers."""
        self.K_normalizer = RollingNormalizer(window_size=self.window_size)
        self.S_normalizer = RollingNormalizer(window_size=self.window_size)
        self.D_normalizer = RollingNormalizer(window_size=self.window_size)

    def normalize(self, signals: CutScoreSignals) -> NormalizedSignals:
        """Normalize raw signals using rolling z-score.

        Args:
            signals: Raw signal values.

        Returns:
            Normalized signal values.
        """
        return NormalizedSignals(
            K_norm=self.K_normalizer.update(signals.K),
            S_norm=self.S_normalizer.update(signals.S),
            D_norm=self.D_normalizer.update(signals.D),
            B=signals.B,
            L=signals.L,
        )

    def reset(self) -> None:
        """Reset all normalizers."""
        self.K_normalizer.reset()
        self.S_normalizer.reset()
        self.D_normalizer.reset()


def compute_cut_score(
    signals: CutScoreSignals,
    config: ChunkingConfig,
    normalizers: SignalNormalizers | None = None,
) -> tuple[float, NormalizedSignals]:
    """Compute the cut-score for a position given its signals.

    The cut-score estimates representational instability at a position.
    Higher scores indicate better chunk boundary candidates.

    Args:
        signals: Raw signal values at the current position.
        config: Chunking configuration with weights and thresholds.
        normalizers: Optional pre-initialized normalizers for streaming.
            If None, signals are used without normalization.

    Returns:
        Tuple of (cut_score, normalized_signals).
    """
    # Normalize signals if normalizers provided
    if normalizers is not None:
        norm = normalizers.normalize(signals)
    else:
        # Use raw signals when no normalizers (single-position computation)
        norm = NormalizedSignals(
            K_norm=signals.K,
            S_norm=signals.S,
            D_norm=signals.D,
            B=signals.B,
            L=signals.L,
        )

    # Compute cut-score components
    score = 0.0

    # Curvature term: high curvature -> good boundary
    if config.use_curvature:
        score += config.wK * relu(norm.K_norm - config.k0)

    # Disharmony term: high disharmony -> good boundary
    if config.use_disharmony:
        score += config.wD * relu(norm.D_norm - config.d0)

    # Stability margin term: LOW stability -> good boundary
    # Note: s0 - S because we want boundaries where stability is LOW
    if config.use_stability_margin:
        score += config.wS * relu(config.s0 - norm.S_norm)

    # Structural boundary term: direct contribution
    if config.use_lil_boundaries:
        score += config.wB * norm.B

    # Length penalty term: penalize deviation from target length
    if config.L_target_bytes > 0:
        length_deviation = (norm.L - config.L_target_bytes) / config.L_target_bytes
        score += config.wL * relu(length_deviation)

    return score, norm


def compute_cut_score_simple(
    signals: CutScoreSignals,
    config: ChunkingConfig,
) -> float:
    """Simplified cut-score computation without normalization tracking.

    Convenience function for single-position scoring without streaming context.

    Args:
        signals: Raw signal values at the current position.
        config: Chunking configuration with weights and thresholds.

    Returns:
        The cut-score value.
    """
    score, _ = compute_cut_score(signals, config, normalizers=None)
    return score
