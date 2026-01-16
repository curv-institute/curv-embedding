"""
SQLite metadata store for curv-embedding.

This is the source of truth for chunk metadata and FAISS ID mapping.
Uses WAL mode for concurrent read/write support.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from src.config import StorageConfig


@dataclass
class ChunkRecord:
    """
    Data class representing a chunk record in the database.

    All fields correspond to columns in the chunks table.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    byte_offset_start: int
    byte_offset_end: int
    content_sha256: str

    # FAISS mapping
    faiss_id: int | None = None

    # Embedding metadata
    embedding_checksum: str | None = None
    embedding_model_name: str | None = None
    embedding_model_version: str | None = None

    # Chunking diagnostics
    cut_score: float | None = None
    curvature_signal: float | None = None
    stability_margin_signal: float | None = None
    disharmony_signal: float | None = None
    is_structural_boundary: bool = False

    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChunkRecord:
        """Create ChunkRecord from a database row."""
        return cls(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=row["chunk_index"],
            byte_offset_start=row["byte_offset_start"],
            byte_offset_end=row["byte_offset_end"],
            content_sha256=row["content_sha256"],
            faiss_id=row["faiss_id"],
            embedding_checksum=row["embedding_checksum"],
            embedding_model_name=row["embedding_model_name"],
            embedding_model_version=row["embedding_model_version"],
            cut_score=row["cut_score"],
            curvature_signal=row["curvature_signal"],
            stability_margin_signal=row["stability_margin_signal"],
            disharmony_signal=row["disharmony_signal"],
            is_structural_boundary=bool(row["is_structural_boundary"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class SQLiteStore:
    """
    SQLite-based metadata store for curv-embedding.

    Manages document and chunk metadata, FAISS ID mapping, ingestion runs,
    and event logging. Uses WAL mode for concurrent access.
    """

    def __init__(self, db_path: Path, config: StorageConfig) -> None:
        """
        Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file.
            config: StorageConfig with journal mode, synchronous, and foreign keys settings.
        """
        self._db_path = Path(db_path)
        self._config = config

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, we manage transactions manually
        )
        self._conn.row_factory = sqlite3.Row

        # Configure pragmas
        self._configure_pragmas()

    def _configure_pragmas(self) -> None:
        """Configure SQLite pragmas based on config."""
        cursor = self._conn.cursor()

        # Journal mode (WAL for concurrent access)
        cursor.execute(f"PRAGMA journal_mode={self._config.sqlite_journal_mode}")

        # Synchronous mode
        cursor.execute(f"PRAGMA synchronous={self._config.sqlite_synchronous}")

        # Foreign keys
        fk_value = "ON" if self._config.sqlite_foreign_keys else "OFF"
        cursor.execute(f"PRAGMA foreign_keys={fk_value}")

        cursor.close()

    def initialize_schema(self) -> None:
        """
        Initialize the database schema from schema.sql.

        Reads and executes the schema file to create all tables and indexes.
        """
        schema_path = Path(__file__).parent / "schema.sql"

        with open(schema_path) as f:
            schema_sql = f.read()

        self._conn.executescript(schema_sql)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context manager for database transactions.

        Yields a cursor and commits on success, rolls back on exception.

        Yields:
            sqlite3.Cursor for executing queries within the transaction.
        """
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def add_document(
        self,
        doc_id: str,
        source_path: str,
        content_sha256: str,
        byte_length: int,
        metadata: dict | None = None,
    ) -> None:
        """
        Add a document record to the database.

        Args:
            doc_id: Unique document identifier.
            source_path: Path to the source file.
            content_sha256: SHA256 hash of document content.
            byte_length: Document size in bytes.
            metadata: Optional JSON-serializable metadata dict.
        """
        metadata_json = json.dumps(metadata) if metadata else None

        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (doc_id, source_path, content_sha256, byte_length, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, source_path, content_sha256, byte_length, metadata_json),
            )

    def add_chunk(self, chunk: ChunkRecord) -> None:
        """
        Add a chunk record to the database.

        Args:
            chunk: ChunkRecord dataclass with all chunk fields.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, byte_offset_start, byte_offset_end,
                    content_sha256, faiss_id, embedding_checksum, embedding_model_name,
                    embedding_model_version, cut_score, curvature_signal,
                    stability_margin_signal, disharmony_signal, is_structural_boundary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.byte_offset_start,
                    chunk.byte_offset_end,
                    chunk.content_sha256,
                    chunk.faiss_id,
                    chunk.embedding_checksum,
                    chunk.embedding_model_name,
                    chunk.embedding_model_version,
                    chunk.cut_score,
                    chunk.curvature_signal,
                    chunk.stability_margin_signal,
                    chunk.disharmony_signal,
                    1 if chunk.is_structural_boundary else 0,
                ),
            )

    def update_chunk_faiss_id(self, chunk_id: str, faiss_id: int) -> None:
        """
        Update the FAISS ID for a chunk.

        Args:
            chunk_id: The chunk identifier.
            faiss_id: The FAISS vector ID to associate.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE chunks
                SET faiss_id = ?, updated_at = datetime('now')
                WHERE chunk_id = ?
                """,
                (faiss_id, chunk_id),
            )

    def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding_checksum: str,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Update embedding metadata for a chunk.

        Args:
            chunk_id: The chunk identifier.
            embedding_checksum: SHA256 checksum of the embedding vector.
            model_name: Name of the embedding model used.
            model_version: Version of the embedding model used.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE chunks
                SET embedding_checksum = ?,
                    embedding_model_name = ?,
                    embedding_model_version = ?,
                    updated_at = datetime('now')
                WHERE chunk_id = ?
                """,
                (embedding_checksum, model_name, model_version, chunk_id),
            )

    def get_chunks_by_content_hash(self, content_sha256: str) -> list[ChunkRecord]:
        """
        Get all chunks with a specific content hash.

        Used for drift calculation to find matching chunks across runs.

        Args:
            content_sha256: SHA256 hash of chunk content.

        Returns:
            List of ChunkRecord objects with matching content hash.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT * FROM chunks WHERE content_sha256 = ?
            """,
            (content_sha256,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [ChunkRecord.from_row(row) for row in rows]

    def get_all_chunks(self) -> list[ChunkRecord]:
        """
        Get all chunks from the database.

        Returns:
            List of all ChunkRecord objects.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index")
        rows = cursor.fetchall()
        cursor.close()
        return [ChunkRecord.from_row(row) for row in rows]

    def log_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        details: dict,
        run_id: str | None = None,
    ) -> None:
        """
        Log an event to the events table.

        Args:
            event_type: Type of event (e.g., document_added, chunk_created).
            entity_type: Type of entity (e.g., document, chunk, index).
            entity_id: Identifier of the entity.
            details: JSON-serializable dictionary with event details.
            run_id: Optional ingestion run ID.
        """
        details_json = json.dumps(details)

        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO events (run_id, event_type, entity_type, entity_id, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, event_type, entity_type, entity_id, details_json),
            )

    def start_ingestion_run(
        self,
        run_id: str,
        config_hash: str,
        config_snapshot: dict,
        seed: int,
    ) -> None:
        """
        Start a new ingestion run.

        Args:
            run_id: Unique identifier for the run.
            config_hash: Hash of the configuration.
            config_snapshot: Full configuration as dictionary.
            seed: Random seed used for the run.
        """
        config_json = json.dumps(config_snapshot)

        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_runs (run_id, config_hash, config_snapshot, seed, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (run_id, config_hash, config_json, seed),
            )

    def complete_ingestion_run(
        self,
        run_id: str,
        metrics_summary: dict | None = None,
    ) -> None:
        """
        Mark an ingestion run as completed.

        Args:
            run_id: The run identifier.
            metrics_summary: Optional summary metrics dictionary.
        """
        metrics_json = json.dumps(metrics_summary) if metrics_summary else None

        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE ingestion_runs
                SET completed_at = datetime('now'),
                    status = 'completed',
                    metrics_summary = ?
                WHERE run_id = ?
                """,
                (metrics_json, run_id),
            )

    def checkpoint(self) -> None:
        """
        Force a WAL checkpoint to flush writes to the main database file.

        Executes PRAGMA wal_checkpoint(FULL) to ensure all WAL content
        is written to the main database file.
        """
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.close()

    def close(self) -> None:
        """
        Close the database connection.

        Performs a checkpoint before closing to ensure all data is flushed.
        """
        if self._conn:
            self.checkpoint()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteStore:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, closing the connection."""
        self.close()
