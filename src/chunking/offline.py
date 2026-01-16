"""
Offline chunking implementation.

Performs full-document scanning to compute cut-scores at each position
and selects optimal chunk boundaries based on local maxima, subject to
size constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.chunking.cut_score import (
    CutScoreSignals,
    NormalizedSignals,
    SignalNormalizers,
    compute_cut_score,
)

if TYPE_CHECKING:
    from src.config import ChunkingConfig


@dataclass
class Chunk:
    """A chunk of content with associated metadata.

    Attributes:
        byte_start: Starting byte offset in the original document (inclusive).
        byte_end: Ending byte offset in the original document (exclusive).
        content: The chunk content as bytes.
        cut_score: The cut-score at the boundary where this chunk ends.
        signals: The raw signals at the chunk boundary.
        normalized_signals: The z-score normalized signals at the boundary.
    """

    byte_start: int
    byte_end: int
    content: bytes
    cut_score: float
    signals: CutScoreSignals
    normalized_signals: NormalizedSignals


def _compute_byte_entropy(data: bytes, start: int, window: int) -> float:
    """Compute Shannon entropy of bytes in a window.

    Used as a proxy for curvature signal in v0.1.

    Args:
        data: The full document bytes.
        start: Start position of the window.
        window: Window size in bytes.

    Returns:
        Shannon entropy in bits (0 to 8 for bytes).
    """
    end = min(start + window, len(data))
    if end <= start:
        return 0.0

    window_bytes = data[start:end]
    if len(window_bytes) == 0:
        return 0.0

    # Count byte frequencies
    counts: dict[int, int] = {}
    for b in window_bytes:
        counts[b] = counts.get(b, 0) + 1

    # Compute entropy
    total = len(window_bytes)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _compute_byte_variance(data: bytes, start: int, window: int) -> float:
    """Compute variance of byte values in a window.

    Used to derive stability margin in v0.1 (stability = 1 / (1 + variance)).

    Args:
        data: The full document bytes.
        start: Start position of the window.
        window: Window size in bytes.

    Returns:
        Variance of byte values.
    """
    end = min(start + window, len(data))
    if end <= start:
        return 0.0

    window_bytes = data[start:end]
    if len(window_bytes) < 2:
        return 0.0

    # Compute mean
    total = sum(window_bytes)
    mean = total / len(window_bytes)

    # Compute variance
    variance = sum((b - mean) ** 2 for b in window_bytes) / len(window_bytes)

    return variance


def _is_structural_boundary(data: bytes, pos: int) -> bool:
    """Check if position is at a structural boundary.

    In v0.1, structural boundaries are simply newlines.

    Args:
        data: The full document bytes.
        pos: Position to check.

    Returns:
        True if position is at a structural boundary.
    """
    if pos >= len(data):
        return False
    # Boundary AFTER a newline character
    return data[pos] == ord("\n")


def _compute_signals_at_position(
    data: bytes,
    pos: int,
    chunk_start: int,
    signal_window: int = 64,
) -> CutScoreSignals:
    """Compute all signals at a given position.

    Args:
        data: The full document bytes.
        pos: Current position.
        chunk_start: Start of current chunk (for length calculation).
        signal_window: Window size for signal computation.

    Returns:
        CutScoreSignals at the position.
    """
    # K: curvature proxy via entropy (higher entropy = more "strain")
    K = _compute_byte_entropy(data, max(0, pos - signal_window // 2), signal_window)

    # S: stability margin proxy via inverse variance
    variance = _compute_byte_variance(
        data, max(0, pos - signal_window // 2), signal_window
    )
    # Normalize to roughly 0-8 range to match entropy scale
    # Using sigmoid-like transform for bounded output
    S = 8.0 / (1.0 + variance / 1000.0)

    # D: disharmony - disabled in v0.1
    D = 0.0

    # B: structural boundary
    B = 1.0 if _is_structural_boundary(data, pos) else 0.0

    # L: chunk length
    L = pos - chunk_start

    return CutScoreSignals(K=K, S=S, D=D, B=B, L=L)


def _find_local_maxima(
    scores: list[tuple[int, float, CutScoreSignals, NormalizedSignals]],
    min_distance: int,
) -> list[tuple[int, float, CutScoreSignals, NormalizedSignals]]:
    """Find local maxima in cut-scores with minimum distance constraint.

    Args:
        scores: List of (position, score, signals, norm_signals) tuples.
        min_distance: Minimum distance between maxima.

    Returns:
        List of local maxima tuples.
    """
    if not scores:
        return []

    maxima: list[tuple[int, float, CutScoreSignals, NormalizedSignals]] = []

    for i, (pos, score, signals, norm) in enumerate(scores):
        is_maximum = True

        # Check neighbors within window
        for j, (other_pos, other_score, _, _) in enumerate(scores):
            if i == j:
                continue
            if abs(pos - other_pos) <= min_distance:
                if other_score > score:
                    is_maximum = False
                    break
                # Tie-breaker: prefer later position
                if other_score == score and other_pos > pos:
                    is_maximum = False
                    break

        if is_maximum:
            maxima.append((pos, score, signals, norm))

    return maxima


def chunk_offline(
    data: bytes,
    config: ChunkingConfig,
    signal_window: int = 64,
) -> list[Chunk]:
    """Chunk a document using offline (full-scan) algorithm.

    Scans the entire document to compute cut-scores at each position,
    then selects optimal boundaries based on local maxima subject to
    size constraints.

    Args:
        data: The document content as bytes.
        config: Chunking configuration.
        signal_window: Window size for signal computation (default 64).

    Returns:
        List of Chunk objects covering the entire document.
    """
    if len(data) == 0:
        return []

    # If document is smaller than min_bytes, return as single chunk
    if len(data) <= config.min_bytes:
        signals = CutScoreSignals(K=0.0, S=0.0, D=0.0, B=0.0, L=len(data))
        norm = NormalizedSignals(K_norm=0.0, S_norm=0.0, D_norm=0.0, B=0.0, L=len(data))
        return [
            Chunk(
                byte_start=0,
                byte_end=len(data),
                content=data,
                cut_score=0.0,
                signals=signals,
                normalized_signals=norm,
            )
        ]

    # Initialize normalizers
    normalizers = SignalNormalizers(window_size=config.commit_horizon_bytes)

    # First pass: compute cut-scores at all positions
    all_scores: list[tuple[int, float, CutScoreSignals, NormalizedSignals]] = []

    # We only consider positions >= min_bytes from chunk start
    # and we need to track chunk starts dynamically
    # For offline, we'll do a simpler approach: compute scores for all positions
    # then select boundaries greedily

    current_chunk_start = 0

    for pos in range(len(data)):
        signals = _compute_signals_at_position(
            data, pos, current_chunk_start, signal_window
        )
        score, norm = compute_cut_score(signals, config, normalizers)
        all_scores.append((pos, score, signals, norm))

    # Select boundaries using greedy algorithm with constraints
    chunks: list[Chunk] = []
    current_start = 0

    while current_start < len(data):
        # If remaining data is <= max_bytes, emit final chunk
        remaining = len(data) - current_start
        if remaining <= config.max_bytes:
            # Find best boundary in remaining range (or just use end)
            end_pos = len(data)
            best_score = 0.0
            best_signals = CutScoreSignals(K=0.0, S=0.0, D=0.0, B=0.0, L=remaining)
            best_norm = NormalizedSignals(
                K_norm=0.0, S_norm=0.0, D_norm=0.0, B=0.0, L=remaining
            )

            # Get the best score from within this final chunk for metadata
            for pos, score, signals, norm in all_scores:
                if current_start < pos <= len(data) and pos >= current_start + config.min_bytes:
                    if score > best_score:
                        best_score = score
                        best_signals = signals
                        best_norm = norm

            chunks.append(
                Chunk(
                    byte_start=current_start,
                    byte_end=end_pos,
                    content=data[current_start:end_pos],
                    cut_score=best_score,
                    signals=best_signals,
                    normalized_signals=best_norm,
                )
            )
            break

        # Find valid boundary candidates within [min_bytes, max_bytes] from current_start
        min_pos = current_start + config.min_bytes
        max_pos = min(current_start + config.max_bytes, len(data))

        candidates = [
            (pos, score, signals, norm)
            for pos, score, signals, norm in all_scores
            if min_pos <= pos <= max_pos
        ]

        if not candidates:
            # No valid candidates - force boundary at max_bytes
            end_pos = max_pos
            signals = _compute_signals_at_position(
                data, end_pos, current_start, signal_window
            )
            norm = NormalizedSignals(
                K_norm=0.0,
                S_norm=0.0,
                D_norm=0.0,
                B=signals.B,
                L=end_pos - current_start,
            )
            chunks.append(
                Chunk(
                    byte_start=current_start,
                    byte_end=end_pos,
                    content=data[current_start:end_pos],
                    cut_score=0.0,
                    signals=signals,
                    normalized_signals=norm,
                )
            )
            current_start = end_pos - config.overlap_bytes
            continue

        # Find local maxima among candidates
        maxima = _find_local_maxima(candidates, min_distance=config.min_bytes // 4)

        if maxima:
            # Select the highest scoring maximum
            best = max(maxima, key=lambda x: x[1])
        else:
            # Fall back to highest scoring candidate
            best = max(candidates, key=lambda x: x[1])

        end_pos, score, signals, norm = best

        chunks.append(
            Chunk(
                byte_start=current_start,
                byte_end=end_pos,
                content=data[current_start:end_pos],
                cut_score=score,
                signals=signals,
                normalized_signals=norm,
            )
        )

        # Advance start position with overlap
        if config.overlap_bytes > 0 and end_pos < len(data):
            current_start = end_pos - config.overlap_bytes
        else:
            current_start = end_pos

    return chunks


def chunk_offline_no_overlap(
    data: bytes,
    config: ChunkingConfig,
    signal_window: int = 64,
) -> list[Chunk]:
    """Chunk a document without any overlap between chunks.

    Convenience function that temporarily disables overlap.

    Args:
        data: The document content as bytes.
        config: Chunking configuration.
        signal_window: Window size for signal computation.

    Returns:
        List of non-overlapping Chunk objects.
    """
    # Create a modified config with no overlap
    from dataclasses import replace

    no_overlap_config = replace(config, overlap_bytes=0)
    return chunk_offline(data, no_overlap_config, signal_window)
