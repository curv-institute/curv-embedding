"""
Wrapper for sentence-transformers embedding model.

Provides deterministic, version-pinned embedding generation for curv-embedding.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from src.config import EmbeddingConfig


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.

    Provides batch and single-text embedding with deterministic behavior,
    normalization support, and embedding checksumming for reproducibility.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialize the embedding model.

        Args:
            config: EmbeddingConfig with model_name, embedding_dim, batch_size, normalize.
        """
        self._config = config
        self._model_name = config.model_name
        self._model_version = config.model_version
        self._embedding_dim = config.embedding_dim
        self._batch_size = config.batch_size
        self._normalize = config.normalize

        # Load the model
        self._model = SentenceTransformer(self._model_name)

        # Verify embedding dimension matches expected
        test_embedding = self._model.encode(["test"], convert_to_numpy=True)
        actual_dim = test_embedding.shape[1]
        if actual_dim != self._embedding_dim:
            raise ValueError(
                f"Model {self._model_name} produces embeddings of dimension {actual_dim}, "
                f"but config specifies {self._embedding_dim}"
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with float32 embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Single text string to embed.

        Returns:
            np.ndarray of shape (embedding_dim,) with float32 embedding.
        """
        embeddings = self.embed([text])
        return embeddings[0]

    def model_info(self) -> dict:
        """
        Return model metadata.

        Returns:
            Dictionary with model name, version, and embedding dimension.
        """
        return {
            "model_name": self._model_name,
            "model_version": self._model_version,
            "embedding_dim": self._embedding_dim,
            "normalize": self._normalize,
            "batch_size": self._batch_size,
        }

    @staticmethod
    def embedding_checksum(embedding: np.ndarray) -> str:
        """
        Compute SHA256 checksum of an embedding vector.

        Args:
            embedding: np.ndarray embedding vector.

        Returns:
            SHA256 hex digest of the embedding bytes.
        """
        # Ensure consistent byte representation
        embedding_bytes = embedding.astype(np.float32).tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()
