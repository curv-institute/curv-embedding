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
Main experiment runner for stability-driven chunking evaluation.

Runs a complete experiment comparing baseline vs stability-driven chunking:
1. Generate or load synthetic data
2. Chunk documents using both methods
3. Embed and index chunks
4. Evaluate metrics (drift, churn, overlap, maintenance)
5. Generate output artifacts

Usage:
    uv run scripts/run_experiment.py --run-name my_experiment
    uv run scripts/run_experiment.py --run-name my_experiment --config configs/default.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config
from src.data.generator import generate_corpus
from src.data.manifests import generate_data_manifest, generate_query_families
from src.chunking.offline import chunk_offline
from src.chunking.manifests import generate_manifest
from src.embedding.model import EmbeddingModel
from src.storage.sqlite_store import SQLiteStore, ChunkRecord
from src.storage.faiss_index import FAISSIndex
from src.eval.drift import compute_drift_stats
from src.eval.churn import compute_churn_stats
from src.eval.overlap import compute_overlap_stats
from src.eval.maintenance import compute_maintenance_stats


def setup_logging(log_level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_git_version() -> dict[str, str | None]:
    """Get current git tag and commit hash for traceability."""
    version_info = {"tag": None, "commit": None, "dirty": False}

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        version_info["commit"] = result.stdout.strip()[:12]

        # Get tag if on a tagged commit
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version_info["tag"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        version_info["dirty"] = bool(result.stdout.strip())

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return version_info


def validate_run_name(run_name: str, git_version: dict) -> None:
    """Warn if run name doesn't follow naming convention."""
    logger = logging.getLogger(__name__)

    # Check if run name contains a version pattern
    version_pattern = r"v\d+\.\d+\.\d+"
    has_version = bool(re.search(version_pattern, run_name))

    if not has_version:
        tag = git_version.get("tag") or f"commit-{git_version.get('commit', 'unknown')}"
        logger.warning(
            f"Run name '{run_name}' does not include version tag. "
            f"Recommended: '{tag}-{run_name}' or '{run_name}_{tag}'"
        )

    if git_version.get("dirty"):
        logger.warning(
            "Working directory has uncommitted changes. "
            "Results may not be reproducible from the recorded commit."
        )


def generate_run_id(config: Config) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_hash = config.config_hash()[:8]
    return f"{timestamp}_{config_hash}"


def setup_run_directory(output_dir: Path, run_name: str) -> Path:
    """Create run directory structure."""
    run_dir = output_dir / run_name
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    return run_dir


def baseline_chunk_offline(data: bytes, config: Config) -> list[dict]:
    """Fixed-size baseline chunking."""
    chunk_size = config.baseline.fixed_chunk_bytes
    overlap = config.baseline.fixed_overlap_bytes
    step = chunk_size - overlap

    chunks = []
    offset = 0
    idx = 0

    while offset < len(data):
        end = min(offset + chunk_size, len(data))
        content = data[offset:end]
        content_hash = hashlib.sha256(content).hexdigest()

        chunks.append({
            "chunk_index": idx,
            "byte_offset_start": offset,
            "byte_offset_end": end,
            "content": content,
            "content_sha256": content_hash,
            "cut_score": 0.0,
            "signals": {"method": "fixed"},
        })

        offset += step
        idx += 1

        if offset >= len(data) and end == len(data):
            break

    return chunks


def run_experiment(
    run_name: str,
    config: Config,
    data_dir: Path | None = None,
) -> dict:
    """Run complete experiment."""
    logger = logging.getLogger(__name__)

    # Get git version for traceability
    git_version = get_git_version()

    # Validate run name follows convention
    validate_run_name(run_name, git_version)

    # Setup
    output_dir = Path(config.general.output_dir)
    run_dir = setup_run_directory(output_dir, run_name)
    run_id = generate_run_id(config)

    logger.info(f"Starting experiment run: {run_name} (ID: {run_id})")
    logger.info(f"Git version: tag={git_version.get('tag')}, commit={git_version.get('commit')}")
    logger.info(f"Output directory: {run_dir}")

    # Save config
    config_path = run_dir / "manifests" / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    metrics_file = run_dir / "metrics.jsonl"

    # Generate or load data
    if data_dir and data_dir.exists():
        logger.info(f"Loading data from {data_dir}")
        # Load existing data
        with open(data_dir / "manifest.json") as f:
            data_manifest = json.load(f)
        documents = []
        corpus_dir = data_dir / "corpus"
        for doc_id in data_manifest.get("document_ids", []):
            # Find the file
            for ext in [".txt", ".py", ".json", ".log"]:
                fpath = corpus_dir / f"{doc_id}{ext}"
                if fpath.exists():
                    content = fpath.read_bytes()
                    documents.append({
                        "doc_id": doc_id,
                        "content": content,
                        "domain": ext[1:] if ext != ".txt" else "text",
                    })
                    break
    else:
        logger.info("Generating synthetic data")
        corpus = generate_corpus(
            seed=config.general.seed,
            num_docs=20,  # Small corpus for v0.1
            domains=["text", "code", "json", "logs"],
            size_range=(1024, 8192),
        )
        documents = [
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "domain": doc.domain,
                "boundary_offsets": doc.boundary_offsets,
            }
            for doc in corpus
        ]
        data_manifest = generate_data_manifest(corpus, config.general.seed, config)

    logger.info(f"Processing {len(documents)} documents")

    # Initialize embedding model
    logger.info("Loading embedding model")
    embed_model = EmbeddingModel(config.embedding)
    model_info = embed_model.model_info()

    # Process with both methods
    results = {"baseline": {}, "stability": {}}

    for method in ["baseline", "stability"]:
        logger.info(f"Processing with {method} chunking")

        # Setup storage
        db_path = run_dir / f"meta_{method}.sqlite"
        index_path = run_dir / f"index_{method}.faiss"

        store = SQLiteStore(db_path, config.storage)
        store.initialize_schema()
        store.start_ingestion_run(
            run_id=f"{run_id}_{method}",
            config_hash=config.config_hash(),
            config_snapshot=config.to_dict(),
            seed=config.general.seed,
        )

        faiss_index = FAISSIndex(config.embedding.embedding_dim, config.storage)

        all_chunks = []
        all_embeddings = {}

        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            content_hash = hashlib.sha256(content).hexdigest()

            # Add document
            store.add_document(
                doc_id=doc_id,
                source_path=f"corpus/{doc_id}",
                content_sha256=content_hash,
                byte_length=len(content),
            )

            # Chunk
            if method == "baseline":
                chunks = baseline_chunk_offline(content, config)
            else:
                chunk_objs = chunk_offline(content, config.chunking)
                chunks = [
                    {
                        "chunk_index": i,
                        "byte_offset_start": c.byte_start,
                        "byte_offset_end": c.byte_end,
                        "content": c.content,
                        "content_sha256": hashlib.sha256(c.content).hexdigest(),
                        "cut_score": c.cut_score,
                        "signals": {
                            "K": c.signals.K if c.signals else 0.0,
                            "S": c.signals.S if c.signals else 0.0,
                            "D": c.signals.D if c.signals else 0.0,
                            "B": c.signals.B if c.signals else 0.0,
                        },
                    }
                    for i, c in enumerate(chunk_objs)
                ]

            # Embed and store chunks
            for chunk_data in chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_data['chunk_index']}"

                # Embed
                text = chunk_data["content"].decode("utf-8", errors="replace")
                embedding = embed_model.embed_single(text)
                embedding_checksum = embed_model.embedding_checksum(embedding)

                # Add to FAISS
                faiss_ids = faiss_index.add_vectors(embedding.reshape(1, -1))
                faiss_id = faiss_ids[0]

                # Store in SQLite
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_data["chunk_index"],
                    byte_offset_start=chunk_data["byte_offset_start"],
                    byte_offset_end=chunk_data["byte_offset_end"],
                    content_sha256=chunk_data["content_sha256"],
                    faiss_id=faiss_id,
                    embedding_checksum=embedding_checksum,
                    embedding_model_name=model_info["model_name"],
                    embedding_model_version=model_info["model_version"],
                    cut_score=chunk_data.get("cut_score", 0.0),
                )
                store.add_chunk(record)

                all_chunks.append(chunk_data)
                all_embeddings[chunk_data["content_sha256"]] = embedding

                store.log_event(
                    event_type="chunk_created",
                    entity_type="chunk",
                    entity_id=chunk_id,
                    details={"doc_id": doc_id, "method": method},
                    run_id=f"{run_id}_{method}",
                )

        # Save artifacts
        faiss_index.save(index_path)
        store.complete_ingestion_run(f"{run_id}_{method}")
        store.checkpoint()
        store.close()

        results[method] = {
            "chunks": all_chunks,
            "embeddings": all_embeddings,
            "num_chunks": len(all_chunks),
            "faiss_path": str(index_path),
            "db_path": str(db_path),
        }

        # Log metrics
        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "metric": "chunk_count",
                "value": len(all_chunks),
            }) + "\n")

            sizes = [c["byte_offset_end"] - c["byte_offset_start"] for c in all_chunks]
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "metric": "mean_chunk_size",
                "value": sum(sizes) / len(sizes) if sizes else 0,
            }) + "\n")

    # Compute comparison metrics
    logger.info("Computing evaluation metrics")

    # Drift (comparing identical content across methods)
    common_hashes = set(results["baseline"]["embeddings"].keys()) & set(results["stability"]["embeddings"].keys())
    if common_hashes:
        baseline_emb = {h: results["baseline"]["embeddings"][h] for h in common_hashes}
        stability_emb = {h: results["stability"]["embeddings"][h] for h in common_hashes}
        drift_results = compute_drift_stats(baseline_emb, stability_emb)

        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric": "drift_comparison",
                "data": drift_results.to_dict(),
            }) + "\n")

    # Maintenance cost
    baseline_hashes = set(results["baseline"]["embeddings"].keys())
    stability_hashes = set(results["stability"]["embeddings"].keys())
    maintenance = compute_maintenance_stats(
        baseline_hashes,
        stability_hashes,
        max(len(baseline_hashes), len(stability_hashes)),
    )

    with open(metrics_file, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": "maintenance_comparison",
            "data": maintenance.to_dict(),
        }) + "\n")

    # Summary
    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "config_hash": config.config_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "num_chunks": results["baseline"]["num_chunks"],
        },
        "stability": {
            "num_chunks": results["stability"]["num_chunks"],
        },
        "common_content_hashes": len(common_hashes) if common_hashes else 0,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Manifest
    manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "git_version": {
            "tag": git_version.get("tag"),
            "commit": git_version.get("commit"),
            "dirty": git_version.get("dirty", False),
        },
        "config": config.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.general.seed,
        "artifacts": {
            "baseline_index": str(results["baseline"]["faiss_path"]),
            "baseline_db": str(results["baseline"]["db_path"]),
            "stability_index": str(results["stability"]["faiss_path"]),
            "stability_db": str(results["stability"]["db_path"]),
            "metrics": str(metrics_file),
            "summary": str(run_dir / "summary.json"),
        },
    }

    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Experiment complete. Results in {run_dir}")
    return summary


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run stability-driven chunking experiment"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to pre-generated data directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    config = load_config(args.config) if args.config.exists() else Config()

    try:
        summary = run_experiment(
            run_name=args.run_name,
            config=config,
            data_dir=args.data_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
