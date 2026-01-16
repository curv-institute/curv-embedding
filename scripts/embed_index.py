#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sentence-transformers>=2.2",
#     "faiss-cpu>=1.7",
#     "numpy>=1.26",
# ]
# ///
"""
Embedding and indexing CLI.

Embeds chunks and builds FAISS index with SQLite metadata.

Usage:
    uv run scripts/embed_index.py chunks/ --output index/
    uv run scripts/embed_index.py --manifest manifest.json --output index/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.embedding.model import EmbeddingModel
from src.storage.sqlite_store import SQLiteStore, ChunkRecord
from src.storage.faiss_index import FAISSIndex


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Embed chunks and build index")
    parser.add_argument(
        "chunks_dir",
        type=Path,
        nargs="?",
        help="Directory containing chunk files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest JSON from chunking",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for index and database",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID for tracking (auto-generated if not provided)",
    )

    args = parser.parse_args()

    if not args.chunks_dir and not args.manifest:
        print("Error: Must provide either chunks_dir or --manifest", file=sys.stderr)
        return 1

    config = load_config(args.config) if args.config.exists() else load_config()

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    run_id = args.run_id or f"embed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Initialize components
    print("Loading embedding model...")
    model = EmbeddingModel(config.embedding)
    model_info = model.model_info()

    db_path = args.output / "meta.sqlite"
    index_path = args.output / "index.faiss"

    store = SQLiteStore(db_path, config.storage)
    store.initialize_schema()
    store.start_ingestion_run(
        run_id=run_id,
        config_hash=config.config_hash(),
        config_snapshot=config.to_dict(),
        seed=config.general.seed,
    )

    faiss_index = FAISSIndex(config.embedding.embedding_dim, config.storage)

    # Load chunks
    chunks_data = []

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)

        doc_id = manifest["doc_id"]
        for chunk_meta in manifest["chunks"]:
            chunks_data.append({
                "doc_id": doc_id,
                "chunk_index": chunk_meta["chunk_index"],
                "byte_start": chunk_meta["byte_offset_start"],
                "byte_end": chunk_meta["byte_offset_end"],
                "content_sha256": chunk_meta["content_sha256"],
                "cut_score": chunk_meta.get("cut_score", 0.0),
                # Note: actual content needs to be loaded separately
            })
    else:
        # Load from directory
        for chunk_file in sorted(args.chunks_dir.glob("chunk_*.txt")):
            content = chunk_file.read_bytes()
            idx = int(chunk_file.stem.split("_")[1])
            chunks_data.append({
                "doc_id": "unknown",
                "chunk_index": idx,
                "byte_start": 0,
                "byte_end": len(content),
                "content": content,
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "cut_score": 0.0,
            })

        for chunk_file in sorted(args.chunks_dir.glob("chunk_*.json")):
            with open(chunk_file) as f:
                data = json.load(f)
            content = data["content"].encode("utf-8")
            chunks_data.append({
                "doc_id": "unknown",
                "chunk_index": data["index"],
                "byte_start": data.get("byte_start", 0),
                "byte_end": data.get("byte_end", len(content)),
                "content": content,
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "cut_score": data.get("cut_score", 0.0),
            })

    print(f"Processing {len(chunks_data)} chunks...")

    # Process chunks
    for chunk_data in chunks_data:
        chunk_id = f"{chunk_data['doc_id']}_chunk_{chunk_data['chunk_index']}"

        # Get content for embedding
        if "content" in chunk_data:
            text = chunk_data["content"].decode("utf-8", errors="replace")
        else:
            text = f"[Chunk {chunk_data['chunk_index']} placeholder]"

        # Embed
        embedding = model.embed_single(text)
        embedding_checksum = model.embedding_checksum(embedding)

        # Add to FAISS
        faiss_ids = faiss_index.add_vectors(embedding.reshape(1, -1))
        faiss_id = faiss_ids[0]

        # Store metadata
        record = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=chunk_data["doc_id"],
            chunk_index=chunk_data["chunk_index"],
            byte_offset_start=chunk_data["byte_start"],
            byte_offset_end=chunk_data["byte_end"],
            content_sha256=chunk_data["content_sha256"],
            faiss_id=faiss_id,
            embedding_checksum=embedding_checksum,
            embedding_model_name=model_info["model_name"],
            embedding_model_version=model_info["model_version"],
            cut_score=chunk_data["cut_score"],
        )
        store.add_chunk(record)

        store.log_event(
            event_type="chunk_embedded",
            entity_type="chunk",
            entity_id=chunk_id,
            details={"faiss_id": faiss_id},
            run_id=run_id,
        )

    # Save
    faiss_index.save(index_path)
    store.complete_ingestion_run(run_id)
    store.checkpoint()
    store.close()

    print(f"\nIndex saved to: {index_path}")
    print(f"Database saved to: {db_path}")
    print(f"Total vectors: {faiss_index.num_vectors()}")

    # Write manifest
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash(),
        "num_chunks": len(chunks_data),
        "embedding_model": model_info,
        "artifacts": {
            "index": str(index_path),
            "database": str(db_path),
        },
    }

    with open(args.output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
