"""
Chunk manifest generation.

Provides metadata generation for chunks including content hashes,
byte offsets, cut-scores, and signals for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.chunking.cut_score import CutScoreSignals, NormalizedSignals
from src.chunking.offline import Chunk

if TYPE_CHECKING:
    from src.config import ChunkingConfig, Config


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk.

    Attributes:
        index: Zero-based index of the chunk in the document.
        byte_start: Starting byte offset (inclusive).
        byte_end: Ending byte offset (exclusive).
        byte_length: Length in bytes.
        content_sha256: SHA256 hash of the chunk content.
        cut_score: Cut-score at the chunk boundary.
        signals: Raw signals at the boundary.
        normalized_signals: Z-score normalized signals at the boundary.
    """

    index: int
    byte_start: int
    byte_end: int
    byte_length: int
    content_sha256: str
    cut_score: float
    signals: dict[str, float]
    normalized_signals: dict[str, float]


@dataclass
class ChunkManifest:
    """Complete manifest for a chunked document.

    Attributes:
        doc_id: Unique identifier for the document.
        doc_content_sha256: SHA256 hash of the entire document.
        total_bytes: Total document size in bytes.
        chunk_count: Number of chunks.
        chunks: List of chunk metadata.
        config_hash: Hash of the configuration used.
        config: The chunking configuration parameters.
        created_at: ISO 8601 timestamp of manifest creation.
        version: Manifest format version.
    """

    doc_id: str
    doc_content_sha256: str
    total_bytes: int
    chunk_count: int
    chunks: list[ChunkMetadata]
    config_hash: str
    config: dict[str, Any]
    created_at: str
    version: str = "1.0.0"


def _signals_to_dict(signals: CutScoreSignals) -> dict[str, float]:
    """Convert CutScoreSignals to a dictionary.

    Args:
        signals: Raw signal values.

    Returns:
        Dictionary with signal names as keys.
    """
    return {
        "K": signals.K,
        "S": signals.S,
        "D": signals.D,
        "B": signals.B,
        "L": float(signals.L),
    }


def _normalized_signals_to_dict(norm: NormalizedSignals) -> dict[str, float]:
    """Convert NormalizedSignals to a dictionary.

    Args:
        norm: Normalized signal values.

    Returns:
        Dictionary with signal names as keys.
    """
    return {
        "K_norm": norm.K_norm,
        "S_norm": norm.S_norm,
        "D_norm": norm.D_norm,
        "B": norm.B,
        "L": float(norm.L),
    }


def _compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Hex-encoded SHA256 hash.
    """
    return hashlib.sha256(data).hexdigest()


def _chunking_config_to_dict(config: ChunkingConfig) -> dict[str, Any]:
    """Convert ChunkingConfig to a dictionary.

    Args:
        config: Chunking configuration.

    Returns:
        Dictionary representation.
    """
    return {
        "min_bytes": config.min_bytes,
        "max_bytes": config.max_bytes,
        "overlap_bytes": config.overlap_bytes,
        "commit_horizon_bytes": config.commit_horizon_bytes,
        "L_target_bytes": config.L_target_bytes,
        "wK": config.wK,
        "wD": config.wD,
        "wS": config.wS,
        "wB": config.wB,
        "wL": config.wL,
        "k0": config.k0,
        "d0": config.d0,
        "s0": config.s0,
        "use_lil_boundaries": config.use_lil_boundaries,
        "use_curvature": config.use_curvature,
        "use_disharmony": config.use_disharmony,
        "use_stability_margin": config.use_stability_margin,
        "soft_trigger_threshold": config.soft_trigger_threshold,
        "soft_trigger_sustain_steps": config.soft_trigger_sustain_steps,
    }


def _compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic hash of configuration.

    Args:
        config_dict: Configuration as dictionary.

    Returns:
        Truncated SHA256 hash (16 chars).
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_manifest(
    chunks: list[Chunk],
    doc_id: str,
    config: Config | ChunkingConfig,
    original_content: bytes | None = None,
) -> dict[str, Any]:
    """Generate a manifest dictionary for a list of chunks.

    Args:
        chunks: List of Chunk objects.
        doc_id: Unique identifier for the document.
        config: Full Config or just ChunkingConfig.
        original_content: Original document bytes (optional, for doc hash).
            If not provided, doc hash is computed from chunk contents.

    Returns:
        Manifest as a dictionary suitable for JSON serialization.
    """
    # Extract chunking config
    if hasattr(config, "chunking"):
        chunking_config = config.chunking
    else:
        chunking_config = config

    config_dict = _chunking_config_to_dict(chunking_config)
    config_hash = _compute_config_hash(config_dict)

    # Compute document hash
    if original_content is not None:
        doc_hash = _compute_sha256(original_content)
        total_bytes = len(original_content)
    else:
        # Reconstruct from chunks (assumes no overlap for accurate hash)
        all_content = b"".join(chunk.content for chunk in chunks)
        doc_hash = _compute_sha256(all_content)
        total_bytes = sum(len(chunk.content) for chunk in chunks)

    # Build chunk metadata
    chunk_metadata_list: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        metadata = ChunkMetadata(
            index=i,
            byte_start=chunk.byte_start,
            byte_end=chunk.byte_end,
            byte_length=len(chunk.content),
            content_sha256=_compute_sha256(chunk.content),
            cut_score=chunk.cut_score,
            signals=_signals_to_dict(chunk.signals),
            normalized_signals=_normalized_signals_to_dict(chunk.normalized_signals),
        )
        chunk_metadata_list.append(asdict(metadata))

    # Build manifest
    manifest = ChunkManifest(
        doc_id=doc_id,
        doc_content_sha256=doc_hash,
        total_bytes=total_bytes,
        chunk_count=len(chunks),
        chunks=[ChunkMetadata(**m) for m in chunk_metadata_list],
        config_hash=config_hash,
        config=config_dict,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return asdict(manifest)


def generate_manifest_from_bytes(
    data: bytes,
    doc_id: str,
    config: Config | ChunkingConfig,
) -> dict[str, Any]:
    """Generate a manifest by chunking bytes and creating metadata.

    Convenience function that performs chunking and manifest generation
    in one step.

    Args:
        data: Document content as bytes.
        doc_id: Unique identifier for the document.
        config: Full Config or just ChunkingConfig.

    Returns:
        Manifest as a dictionary suitable for JSON serialization.
    """
    from src.chunking.offline import chunk_offline

    # Extract chunking config
    if hasattr(config, "chunking"):
        chunking_config = config.chunking
    else:
        chunking_config = config

    chunks = chunk_offline(data, chunking_config)
    return generate_manifest(chunks, doc_id, config, original_content=data)


def manifest_to_json(manifest: dict[str, Any], indent: int = 2) -> str:
    """Convert manifest dictionary to JSON string.

    Args:
        manifest: Manifest dictionary.
        indent: JSON indentation level.

    Returns:
        JSON string representation.
    """
    return json.dumps(manifest, indent=indent, sort_keys=False)


def validate_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a manifest dictionary for required fields and consistency.

    Args:
        manifest: Manifest dictionary to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Check required top-level fields
    required_fields = [
        "doc_id",
        "doc_content_sha256",
        "total_bytes",
        "chunk_count",
        "chunks",
        "config_hash",
        "config",
        "created_at",
        "version",
    ]

    for field_name in required_fields:
        if field_name not in manifest:
            errors.append(f"Missing required field: {field_name}")

    if errors:
        return False, errors

    # Validate chunk count matches
    if len(manifest["chunks"]) != manifest["chunk_count"]:
        errors.append(
            f"Chunk count mismatch: declared {manifest['chunk_count']}, "
            f"found {len(manifest['chunks'])}"
        )

    # Validate each chunk
    required_chunk_fields = [
        "index",
        "byte_start",
        "byte_end",
        "byte_length",
        "content_sha256",
        "cut_score",
        "signals",
        "normalized_signals",
    ]

    for i, chunk in enumerate(manifest["chunks"]):
        for field_name in required_chunk_fields:
            if field_name not in chunk:
                errors.append(f"Chunk {i}: missing required field: {field_name}")

        # Validate byte_length consistency
        if "byte_start" in chunk and "byte_end" in chunk and "byte_length" in chunk:
            expected_length = chunk["byte_end"] - chunk["byte_start"]
            if chunk["byte_length"] != expected_length:
                errors.append(
                    f"Chunk {i}: byte_length {chunk['byte_length']} != "
                    f"byte_end - byte_start ({expected_length})"
                )

        # Validate index
        if "index" in chunk and chunk["index"] != i:
            errors.append(f"Chunk {i}: index mismatch, found {chunk['index']}")

    # Validate config hash
    if "config" in manifest and "config_hash" in manifest:
        expected_hash = _compute_config_hash(manifest["config"])
        if manifest["config_hash"] != expected_hash:
            errors.append(
                f"Config hash mismatch: declared {manifest['config_hash']}, "
                f"computed {expected_hash}"
            )

    return len(errors) == 0, errors


def verify_chunk_integrity(
    manifest: dict[str, Any],
    chunks: list[Chunk],
) -> tuple[bool, list[str]]:
    """Verify that chunks match their manifest metadata.

    Args:
        manifest: Manifest dictionary.
        chunks: List of Chunk objects to verify.

    Returns:
        Tuple of (all_valid, list of error messages).
    """
    errors: list[str] = []

    if len(chunks) != manifest["chunk_count"]:
        errors.append(
            f"Chunk count mismatch: manifest has {manifest['chunk_count']}, "
            f"provided {len(chunks)}"
        )
        return False, errors

    for i, (chunk, meta) in enumerate(zip(chunks, manifest["chunks"])):
        # Verify content hash
        actual_hash = _compute_sha256(chunk.content)
        if actual_hash != meta["content_sha256"]:
            errors.append(
                f"Chunk {i}: content hash mismatch - "
                f"expected {meta['content_sha256']}, got {actual_hash}"
            )

        # Verify byte offsets
        if chunk.byte_start != meta["byte_start"]:
            errors.append(
                f"Chunk {i}: byte_start mismatch - "
                f"expected {meta['byte_start']}, got {chunk.byte_start}"
            )

        if chunk.byte_end != meta["byte_end"]:
            errors.append(
                f"Chunk {i}: byte_end mismatch - "
                f"expected {meta['byte_end']}, got {chunk.byte_end}"
            )

        # Verify length
        if len(chunk.content) != meta["byte_length"]:
            errors.append(
                f"Chunk {i}: byte_length mismatch - "
                f"expected {meta['byte_length']}, got {len(chunk.content)}"
            )

    return len(errors) == 0, errors
