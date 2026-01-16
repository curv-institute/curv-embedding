"""
FAISS index wrapper for curv-embedding.

Provides vector similarity search with exact (Flat) index for v0.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from src.config import StorageConfig


class FAISSIndex:
    """
    FAISS index wrapper for vector similarity search.

    Uses Flat index for exact search in v0.1, ensuring deterministic results.
    Supports L2 (Euclidean) distance metric.
    """

    def __init__(self, dim: int, config: StorageConfig) -> None:
        """
        Initialize the FAISS index.

        Args:
            dim: Embedding dimension.
            config: StorageConfig with faiss_index_type and faiss_metric settings.
        """
        self._dim = dim
        self._config = config

        # Create index based on metric type
        if config.faiss_metric == "L2":
            self._index = faiss.IndexFlatL2(dim)
        elif config.faiss_metric == "IP":
            # Inner product (cosine similarity for normalized vectors)
            self._index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported FAISS metric: {config.faiss_metric}")

        # Track the next ID to assign (Flat index uses sequential IDs starting at 0)
        self._next_id = 0

    def add_vectors(self, vectors: np.ndarray) -> list[int]:
        """
        Add vectors to the index.

        Args:
            vectors: np.ndarray of shape (n, dim) containing vectors to add.

        Returns:
            List of FAISS IDs assigned to the vectors (sequential integers).

        Raises:
            ValueError: If vector dimension doesn't match index dimension.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self._dim}"
            )

        # Ensure float32 for FAISS
        vectors = vectors.astype(np.float32)

        # Add to index
        self._index.add(vectors)

        # Assign sequential IDs
        n_vectors = vectors.shape[0]
        ids = list(range(self._next_id, self._next_id + n_vectors))
        self._next_id += n_vectors

        return ids

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of the query vector.

        Args:
            query: np.ndarray of shape (dim,) or (n, dim) for batch queries.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, ids):
                - distances: np.ndarray of shape (n, k) with distances to neighbors.
                - ids: np.ndarray of shape (n, k) with FAISS IDs of neighbors.

        Raises:
            ValueError: If k exceeds number of indexed vectors.
        """
        if self._index.ntotal == 0:
            # Return empty results for empty index
            if query.ndim == 1:
                return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
            return (
                np.zeros((query.shape[0], 0), dtype=np.float32),
                np.zeros((query.shape[0], 0), dtype=np.int64),
            )

        # Clamp k to available vectors
        k = min(k, self._index.ntotal)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Ensure float32 for FAISS
        query = query.astype(np.float32)

        distances, ids = self._index.search(query, k)

        return distances, ids

    def get_neighbors(self, vector_id: int, k: int) -> list[int]:
        """
        Get k nearest neighbors of an indexed vector.

        Args:
            vector_id: FAISS ID of the vector to find neighbors for.
            k: Number of neighbors to return.

        Returns:
            List of FAISS IDs of the k nearest neighbors (excluding the query vector).

        Raises:
            ValueError: If vector_id is not in the index.
        """
        if vector_id < 0 or vector_id >= self._index.ntotal:
            raise ValueError(
                f"Vector ID {vector_id} not in index (size: {self._index.ntotal})"
            )

        # Reconstruct the vector from the index
        vector = self._index.reconstruct(vector_id)
        vector = vector.reshape(1, -1)

        # Search for k+1 neighbors (including the query vector itself)
        k_search = min(k + 1, self._index.ntotal)
        distances, ids = self._index.search(vector, k_search)

        # Filter out the query vector and return up to k neighbors
        neighbors = [int(vid) for vid in ids[0] if vid != vector_id]
        return neighbors[:k]

    def save(self, path: Path) -> None:
        """
        Save the index to a file.

        Args:
            path: Path to save the index file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

        # Also save the next_id counter for restoration
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "w") as f:
            f.write(str(self._next_id))

    def load(self, path: Path) -> None:
        """
        Load the index from a file.

        Args:
            path: Path to the index file.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self._index = faiss.read_index(str(path))

        # Restore the next_id counter
        meta_path = path.with_suffix(".meta")
        if meta_path.exists():
            with open(meta_path) as f:
                self._next_id = int(f.read().strip())
        else:
            # Fallback: assume sequential IDs from 0
            self._next_id = self._index.ntotal

    def num_vectors(self) -> int:
        """
        Return the number of vectors in the index.

        Returns:
            Number of indexed vectors.
        """
        return self._index.ntotal
