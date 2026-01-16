"""
Streaming chunking implementation.

Processes data incrementally as it arrives, maintaining a trailing buffer
and committing chunks when trigger conditions are met.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from src.chunking.cut_score import (
    CutScoreSignals,
    NormalizedSignals,
    SignalNormalizers,
    compute_cut_score,
)
from src.chunking.offline import Chunk

if TYPE_CHECKING:
    from src.config import ChunkingConfig


@dataclass
class BoundaryCandidate:
    """A potential chunk boundary with its metadata.

    Attributes:
        position: Byte position in the current buffer.
        global_position: Byte position in the overall stream.
        score: Cut-score at this position.
        signals: Raw signals at this position.
        normalized_signals: Normalized signals at this position.
    """

    position: int
    global_position: int
    score: float
    signals: CutScoreSignals
    normalized_signals: NormalizedSignals


@dataclass
class StreamingChunkerState:
    """Internal state for the streaming chunker.

    Attributes:
        buffer: Current accumulated bytes.
        global_offset: Byte offset of buffer start in the overall stream.
        chunk_start_offset: Global byte offset where current chunk started.
        normalizers: Signal normalizers for z-score computation.
        best_boundary: Best boundary candidate within the commit horizon.
        soft_trigger_count: Number of consecutive steps above soft threshold.
        candidates: Recent boundary candidates in the commit horizon.
    """

    buffer: bytearray = field(default_factory=bytearray)
    global_offset: int = 0
    chunk_start_offset: int = 0
    normalizers: SignalNormalizers = field(
        default_factory=lambda: SignalNormalizers(window_size=1024)
    )
    best_boundary: BoundaryCandidate | None = None
    soft_trigger_count: int = 0
    candidates: deque[BoundaryCandidate] = field(
        default_factory=lambda: deque(maxlen=256)
    )


def _compute_byte_entropy_buffer(buffer: bytearray, start: int, window: int) -> float:
    """Compute Shannon entropy of bytes in a buffer window.

    Args:
        buffer: The byte buffer.
        start: Start position of the window.
        window: Window size in bytes.

    Returns:
        Shannon entropy in bits (0 to 8 for bytes).
    """
    end = min(start + window, len(buffer))
    if end <= start:
        return 0.0

    window_bytes = buffer[start:end]
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


def _compute_byte_variance_buffer(buffer: bytearray, start: int, window: int) -> float:
    """Compute variance of byte values in a buffer window.

    Args:
        buffer: The byte buffer.
        start: Start position of the window.
        window: Window size in bytes.

    Returns:
        Variance of byte values.
    """
    end = min(start + window, len(buffer))
    if end <= start:
        return 0.0

    window_bytes = buffer[start:end]
    if len(window_bytes) < 2:
        return 0.0

    # Compute mean
    total = sum(window_bytes)
    mean = total / len(window_bytes)

    # Compute variance
    variance = sum((b - mean) ** 2 for b in window_bytes) / len(window_bytes)

    return variance


def _compute_signals_at_buffer_position(
    buffer: bytearray,
    pos: int,
    chunk_length: int,
    signal_window: int = 64,
) -> CutScoreSignals:
    """Compute all signals at a given buffer position.

    Args:
        buffer: The byte buffer.
        pos: Current position in buffer.
        chunk_length: Current chunk length from start.
        signal_window: Window size for signal computation.

    Returns:
        CutScoreSignals at the position.
    """
    # K: curvature proxy via entropy
    K = _compute_byte_entropy_buffer(
        buffer, max(0, pos - signal_window // 2), signal_window
    )

    # S: stability margin proxy via inverse variance
    variance = _compute_byte_variance_buffer(
        buffer, max(0, pos - signal_window // 2), signal_window
    )
    S = 8.0 / (1.0 + variance / 1000.0)

    # D: disharmony - disabled in v0.1
    D = 0.0

    # B: structural boundary (newline)
    B = 1.0 if (pos < len(buffer) and buffer[pos] == ord("\n")) else 0.0

    return CutScoreSignals(K=K, S=S, D=D, B=B, L=chunk_length)


def chunk_stream(
    data_iter: Iterator[bytes],
    config: ChunkingConfig,
    signal_window: int = 64,
) -> Iterator[Chunk]:
    """Stream chunks from an iterator of byte data.

    Processes data incrementally, maintaining a trailing buffer and
    committing chunks when trigger conditions are met:
    - Hard trigger: chunk length >= max_bytes
    - Soft trigger: cut-score exceeds threshold for sustained steps

    Args:
        data_iter: Iterator yielding bytes chunks.
        config: Chunking configuration.
        signal_window: Window size for signal computation.

    Yields:
        Chunk objects as they are committed.
    """
    state = StreamingChunkerState(
        normalizers=SignalNormalizers(window_size=config.commit_horizon_bytes)
    )

    # Process incoming data
    for data_chunk in data_iter:
        state.buffer.extend(data_chunk)

        # Process buffer and yield any ready chunks
        yield from _process_buffer(state, config, signal_window, is_final=False)

    # Finalize: emit any remaining data as final chunk
    yield from _process_buffer(state, config, signal_window, is_final=True)


def _process_buffer(
    state: StreamingChunkerState,
    config: ChunkingConfig,
    signal_window: int,
    is_final: bool,
) -> Iterator[Chunk]:
    """Process the buffer and yield any ready chunks.

    Args:
        state: Current chunker state.
        config: Chunking configuration.
        signal_window: Window size for signal computation.
        is_final: Whether this is the final processing pass.

    Yields:
        Chunk objects as they are ready.
    """
    while True:
        buffer_len = len(state.buffer)

        if buffer_len == 0:
            break

        # Calculate current chunk length
        chunk_length = buffer_len

        # Hard trigger: must commit at max_bytes
        if chunk_length >= config.max_bytes:
            # Find best boundary within the valid range
            chunk = _commit_chunk_at_best_boundary(state, config, signal_window)
            if chunk is not None:
                yield chunk
            continue

        # If this is final pass and we have data, commit it
        if is_final:
            if buffer_len > 0:
                chunk = _commit_remaining(state, config, signal_window)
                if chunk is not None:
                    yield chunk
            break

        # Check soft trigger conditions if we have enough data
        if chunk_length >= config.min_bytes:
            # Compute signals at current position
            pos = buffer_len - 1
            signals = _compute_signals_at_buffer_position(
                state.buffer, pos, chunk_length, signal_window
            )
            score, norm = compute_cut_score(signals, config, state.normalizers)

            global_pos = state.global_offset + pos

            # Track this as a boundary candidate
            candidate = BoundaryCandidate(
                position=pos,
                global_position=global_pos,
                score=score,
                signals=signals,
                normalized_signals=norm,
            )
            state.candidates.append(candidate)

            # Update best boundary if this is better
            if state.best_boundary is None or score > state.best_boundary.score:
                state.best_boundary = candidate

            # Check soft trigger
            if score >= config.soft_trigger_threshold:
                state.soft_trigger_count += 1

                if state.soft_trigger_count >= config.soft_trigger_sustain_steps:
                    # Soft trigger sustained - commit at best boundary
                    chunk = _commit_at_boundary(
                        state, state.best_boundary, config, signal_window
                    )
                    if chunk is not None:
                        yield chunk
                    continue
            else:
                # Reset soft trigger counter
                state.soft_trigger_count = 0

        # Not ready to commit - need more data
        break


def _commit_chunk_at_best_boundary(
    state: StreamingChunkerState,
    config: ChunkingConfig,
    signal_window: int,
) -> Chunk | None:
    """Commit a chunk at the best boundary within valid range.

    Used for hard trigger when max_bytes is reached.

    Args:
        state: Current chunker state.
        config: Chunking configuration.
        signal_window: Window size for signal computation.

    Returns:
        Committed Chunk or None if buffer is empty.
    """
    if len(state.buffer) == 0:
        return None

    # Find best boundary within [min_bytes, max_bytes]
    min_pos = config.min_bytes
    max_pos = min(config.max_bytes, len(state.buffer))

    best_candidate: BoundaryCandidate | None = None
    best_score = float("-inf")

    # Check recent candidates
    for candidate in state.candidates:
        if min_pos <= candidate.position <= max_pos:
            if candidate.score > best_score:
                best_score = candidate.score
                best_candidate = candidate

    # If no good candidate, scan the range
    if best_candidate is None:
        for pos in range(min_pos, max_pos + 1):
            chunk_length = pos
            signals = _compute_signals_at_buffer_position(
                state.buffer, pos, chunk_length, signal_window
            )
            score, norm = compute_cut_score(signals, config, state.normalizers)

            if score > best_score:
                best_score = score
                best_candidate = BoundaryCandidate(
                    position=pos,
                    global_position=state.global_offset + pos,
                    score=score,
                    signals=signals,
                    normalized_signals=norm,
                )

    # Fall back to max_bytes if still no candidate
    if best_candidate is None:
        pos = max_pos
        chunk_length = pos
        signals = _compute_signals_at_buffer_position(
            state.buffer, pos, chunk_length, signal_window
        )
        score, norm = compute_cut_score(signals, config, state.normalizers)
        best_candidate = BoundaryCandidate(
            position=pos,
            global_position=state.global_offset + pos,
            score=score,
            signals=signals,
            normalized_signals=norm,
        )

    return _commit_at_boundary(state, best_candidate, config, signal_window)


def _commit_at_boundary(
    state: StreamingChunkerState,
    boundary: BoundaryCandidate | None,
    config: ChunkingConfig,
    signal_window: int,
) -> Chunk | None:
    """Commit a chunk at the specified boundary.

    Args:
        state: Current chunker state.
        boundary: Boundary candidate to commit at.
        config: Chunking configuration.
        signal_window: Window size for signal computation.

    Returns:
        Committed Chunk or None if boundary is invalid.
    """
    if boundary is None or len(state.buffer) == 0:
        return None

    end_pos = boundary.position

    # Ensure we don't exceed buffer
    end_pos = min(end_pos, len(state.buffer))

    if end_pos <= 0:
        return None

    # Extract chunk content
    content = bytes(state.buffer[:end_pos])

    # Create chunk
    chunk = Chunk(
        byte_start=state.chunk_start_offset,
        byte_end=state.chunk_start_offset + end_pos,
        content=content,
        cut_score=boundary.score,
        signals=boundary.signals,
        normalized_signals=boundary.normalized_signals,
    )

    # Update state
    overlap = config.overlap_bytes if config.overlap_bytes > 0 else 0
    advance = max(1, end_pos - overlap)

    # Remove committed bytes from buffer (keeping overlap)
    del state.buffer[:advance]
    state.global_offset += advance
    state.chunk_start_offset = state.global_offset

    # Clear candidates and reset
    state.candidates.clear()
    state.best_boundary = None
    state.soft_trigger_count = 0

    return chunk


def _commit_remaining(
    state: StreamingChunkerState,
    config: ChunkingConfig,
    signal_window: int,
) -> Chunk | None:
    """Commit all remaining data as the final chunk.

    Args:
        state: Current chunker state.
        config: Chunking configuration.
        signal_window: Window size for signal computation.

    Returns:
        Final Chunk or None if buffer is empty.
    """
    if len(state.buffer) == 0:
        return None

    pos = len(state.buffer)
    chunk_length = pos

    signals = _compute_signals_at_buffer_position(
        state.buffer, pos - 1, chunk_length, signal_window
    )
    score, norm = compute_cut_score(signals, config, state.normalizers)

    content = bytes(state.buffer)

    chunk = Chunk(
        byte_start=state.chunk_start_offset,
        byte_end=state.chunk_start_offset + pos,
        content=content,
        cut_score=score,
        signals=signals,
        normalized_signals=norm,
    )

    # Clear state
    state.buffer.clear()
    state.global_offset += pos
    state.chunk_start_offset = state.global_offset
    state.candidates.clear()
    state.best_boundary = None
    state.soft_trigger_count = 0

    return chunk


class StreamingChunker:
    """Stateful streaming chunker for incremental processing.

    Provides a class-based interface for streaming chunking when
    the iterator pattern is inconvenient.

    Example:
        chunker = StreamingChunker(config)
        for data in data_source:
            for chunk in chunker.feed(data):
                process(chunk)
        for chunk in chunker.finalize():
            process(chunk)
    """

    def __init__(
        self,
        config: ChunkingConfig,
        signal_window: int = 64,
    ) -> None:
        """Initialize the streaming chunker.

        Args:
            config: Chunking configuration.
            signal_window: Window size for signal computation.
        """
        self.config = config
        self.signal_window = signal_window
        self.state = StreamingChunkerState(
            normalizers=SignalNormalizers(window_size=config.commit_horizon_bytes)
        )
        self._finalized = False

    def feed(self, data: bytes) -> Iterator[Chunk]:
        """Feed data into the chunker and yield any ready chunks.

        Args:
            data: Bytes to process.

        Yields:
            Chunk objects as they are ready.

        Raises:
            RuntimeError: If called after finalize().
        """
        if self._finalized:
            raise RuntimeError("Cannot feed data after finalize()")

        self.state.buffer.extend(data)
        yield from _process_buffer(
            self.state, self.config, self.signal_window, is_final=False
        )

    def finalize(self) -> Iterator[Chunk]:
        """Finalize chunking and yield any remaining data.

        Must be called once when all data has been fed.

        Yields:
            Final Chunk objects.

        Raises:
            RuntimeError: If called more than once.
        """
        if self._finalized:
            raise RuntimeError("finalize() already called")

        self._finalized = True
        yield from _process_buffer(
            self.state, self.config, self.signal_window, is_final=True
        )

    def reset(self) -> None:
        """Reset the chunker state for reuse."""
        self.state = StreamingChunkerState(
            normalizers=SignalNormalizers(window_size=self.config.commit_horizon_bytes)
        )
        self._finalized = False

    @property
    def buffer_size(self) -> int:
        """Return the current buffer size in bytes."""
        return len(self.state.buffer)

    @property
    def total_bytes_processed(self) -> int:
        """Return the total bytes processed so far."""
        return self.state.global_offset + len(self.state.buffer)
