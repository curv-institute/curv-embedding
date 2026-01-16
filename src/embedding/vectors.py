"""
Vector utilities for embedding operations.

Provides distance metrics, normalization, and serialization for embedding vectors.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector (1D np.ndarray).
        b: Second vector (1D np.ndarray).

    Returns:
        Cosine similarity as a float in [-1, 1].

    Raises:
        ValueError: If vectors have different dimensions or are zero vectors.
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero vector")

    return float(np.dot(a, b) / (norm_a * norm_b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two vectors.

    Args:
        a: First vector (1D np.ndarray).
        b: Second vector (1D np.ndarray).

    Returns:
        L2 distance as a non-negative float.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")

    return float(np.linalg.norm(a - b))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors to unit length.

    Args:
        vectors: np.ndarray of shape (n, dim) or (dim,) for single vector.

    Returns:
        L2-normalized vectors with same shape as input.
        Zero vectors remain as zero vectors.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0:
            return vectors.copy()
        return vectors / norm

    # Handle 2D case
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def vectors_to_bytes(vectors: np.ndarray) -> bytes:
    """
    Serialize vectors to bytes for checksumming or storage.

    Uses little-endian float32 format for deterministic serialization.

    Args:
        vectors: np.ndarray of any shape.

    Returns:
        Bytes representation of the vectors.
    """
    # Ensure consistent dtype and byte order
    vectors_f32 = vectors.astype(np.float32)
    return vectors_f32.tobytes()


def bytes_to_vectors(data: bytes, dim: int) -> np.ndarray:
    """
    Deserialize bytes back to vectors.

    Args:
        data: Bytes from vectors_to_bytes().
        dim: Embedding dimension to reshape vectors.

    Returns:
        np.ndarray of shape (n, dim) where n = len(data) / (4 * dim).

    Raises:
        ValueError: If data length is not divisible by (4 * dim).
    """
    if len(data) % (4 * dim) != 0:
        raise ValueError(
            f"Data length {len(data)} not divisible by {4 * dim} "
            f"(4 bytes per float32 * {dim} dimensions)"
        )

    vectors = np.frombuffer(data, dtype=np.float32)
    n_vectors = len(vectors) // dim
    return vectors.reshape(n_vectors, dim)
