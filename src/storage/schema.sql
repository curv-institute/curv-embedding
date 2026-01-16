-- SQLite schema for curv-embedding metadata store
-- This is the source of truth for chunk metadata and FAISS ID mapping

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT,
    content_sha256 TEXT NOT NULL,
    byte_length INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT  -- JSON blob for extensibility
);

-- Chunks table (core metadata)
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,  -- 0-based index within document
    byte_offset_start INTEGER NOT NULL,
    byte_offset_end INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,

    -- FAISS mapping
    faiss_id INTEGER UNIQUE,  -- internal FAISS vector ID

    -- Embedding metadata
    embedding_checksum TEXT,  -- SHA256 of embedding vector
    embedding_model_name TEXT,
    embedding_model_version TEXT,

    -- Chunking diagnostics (from cut-score)
    cut_score REAL,
    curvature_signal REAL,
    stability_margin_signal REAL,
    disharmony_signal REAL,
    is_structural_boundary INTEGER DEFAULT 0,

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT,

    UNIQUE(doc_id, chunk_index)
);

-- Index for content-based matching (drift calculation)
CREATE INDEX IF NOT EXISTS idx_chunks_content_sha256 ON chunks(content_sha256);

-- Index for FAISS ID lookups
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);

-- Index for document lookups
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- Ingestion runs table (for reproducibility)
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    config_hash TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,  -- JSON blob of full config
    seed INTEGER NOT NULL,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    error_message TEXT,
    metrics_summary TEXT  -- JSON blob of summary metrics
);

-- Events table (audit log for all state changes)
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    run_id TEXT REFERENCES ingestion_runs(run_id),
    event_type TEXT NOT NULL,  -- document_added, chunk_created, chunk_updated, etc.
    entity_type TEXT,  -- document, chunk, index
    entity_id TEXT,
    details TEXT  -- JSON blob with event-specific data
);

-- Index for event queries
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id);

-- Query probes table (for evaluation)
CREATE TABLE IF NOT EXISTS query_probes (
    probe_id TEXT PRIMARY KEY,
    family_id TEXT,  -- groups related reformulations
    query_text TEXT NOT NULL,
    query_embedding_checksum TEXT,
    expected_chunk_ids TEXT,  -- JSON array of ground-truth chunk IDs
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for query family lookups
CREATE INDEX IF NOT EXISTS idx_query_probes_family ON query_probes(family_id);

-- Neighbor snapshots table (for churn calculation)
CREATE TABLE IF NOT EXISTS neighbor_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),
    probe_id TEXT NOT NULL,
    k INTEGER NOT NULL,
    neighbor_ids TEXT NOT NULL,  -- JSON array of chunk_ids in rank order
    distances TEXT NOT NULL,  -- JSON array of distances
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for snapshot queries
CREATE INDEX IF NOT EXISTS idx_neighbor_snapshots_run ON neighbor_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_neighbor_snapshots_probe ON neighbor_snapshots(probe_id);
