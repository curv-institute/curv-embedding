#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sentence-transformers>=2.2",
#     "faiss-cpu>=1.7",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "matplotlib>=3.8",
# ]
# ///
"""
Temporal stability experiment for curv-embedding v1.4.0.

Measures embedding drift, ANN neighbor churn, and maintenance cost under
incremental document updates, comparing baseline vs stability-driven chunking.

Update Scenarios:
1. Append-only: Append new content blocks to existing documents
2. Local edits: Small edits within existing documents
3. Boundary stress: Updates near known chunk boundaries

Usage:
    uv run scripts/run_temporal_experiment.py --run-name v1.3.0-baseline-temporal-20260116 --method baseline
    uv run scripts/run_temporal_experiment.py --run-name v1.3.0-stability-temporal-20260116 --method stability
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config
from src.data.generator import generate_corpus, SyntheticDocument
from src.chunking.offline import chunk_offline
from src.embedding.model import EmbeddingModel
from src.storage.sqlite_store import SQLiteStore, ChunkRecord
from src.storage.faiss_index import FAISSIndex
from src.eval.drift import compute_drift_stats, DriftResult
from src.eval.churn import compute_churn_stats, ChurnResult
from src.eval.maintenance import compute_maintenance_stats, MaintenanceResult


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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        version_info["commit"] = result.stdout.strip()[:12]
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            version_info["tag"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True,
        )
        version_info["dirty"] = bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return version_info


@dataclass
class TemporalDocument:
    """A document with temporal versions."""
    doc_id: str
    domain: str
    versions: list[bytes] = field(default_factory=list)  # t0, t1, t2, ...
    version_hashes: list[str] = field(default_factory=list)
    boundary_offsets: list[list[int]] = field(default_factory=list)
    update_scenarios: list[str] = field(default_factory=list)  # append, edit, boundary


@dataclass
class ChunkVersion:
    """A chunk at a specific time step."""
    chunk_id: str
    doc_id: str
    time_step: int
    byte_start: int
    byte_end: int
    content: bytes
    content_sha256: str
    embedding: np.ndarray | None = None
    cut_score: float = 0.0
    near_boundary: bool = False  # within N bytes of original boundary


@dataclass
class TemporalMetrics:
    """Aggregated temporal stability metrics."""
    # Drift metrics (for content-stable chunks)
    drift_cos_mean: float = 0.0
    drift_cos_p90: float = 0.0
    drift_l2_mean: float = 0.0
    drift_l2_p90: float = 0.0
    n_stable_chunks: int = 0

    # Churn metrics
    churn_topk_mean: float = 0.0
    churn_jaccard_mean: float = 0.0

    # Maintenance metrics
    reembed_fraction: float = 0.0
    total_added: int = 0
    total_removed: int = 0
    total_unchanged: int = 0

    # Boundary-localized effects
    drift_near_boundary_mean: float = 0.0
    drift_interior_mean: float = 0.0
    churn_near_boundary_mean: float = 0.0
    churn_interior_mean: float = 0.0

    def to_dict(self) -> dict:
        return {
            "drift_cos_mean": self.drift_cos_mean,
            "drift_cos_p90": self.drift_cos_p90,
            "drift_l2_mean": self.drift_l2_mean,
            "drift_l2_p90": self.drift_l2_p90,
            "n_stable_chunks": self.n_stable_chunks,
            "churn_topk_mean": self.churn_topk_mean,
            "churn_jaccard_mean": self.churn_jaccard_mean,
            "reembed_fraction": self.reembed_fraction,
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "total_unchanged": self.total_unchanged,
            "drift_near_boundary_mean": self.drift_near_boundary_mean,
            "drift_interior_mean": self.drift_interior_mean,
            "churn_near_boundary_mean": self.churn_near_boundary_mean,
            "churn_interior_mean": self.churn_interior_mean,
        }


def generate_append_update(rng: random.Random, content: bytes, domain: str) -> bytes:
    """Generate append-only update: add new content at the end."""
    # Generate 200-800 bytes of new content based on domain
    new_size = rng.randint(200, 800)

    if domain == "text":
        # Generate paragraph-like text
        words = []
        while len(" ".join(words).encode()) < new_size:
            word_len = rng.randint(3, 10)
            word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_len))
            words.append(word)
            if rng.random() < 0.1:
                words.append(".")
        new_content = "\n\n" + " ".join(words)
    elif domain == "code":
        # Generate function-like code
        func_name = "_".join(rng.choice(["get", "set", "update", "process", "handle"])
                            for _ in range(2))
        new_content = f"\n\ndef {func_name}(arg):\n    # New function\n    return arg\n"
    elif domain == "json":
        # Generate JSON object
        new_content = ',\n  {"new_field": "' + "".join(rng.choice("abcdef") for _ in range(20)) + '"}'
    else:  # logs
        # Generate log entries
        lines = []
        for _ in range(rng.randint(5, 15)):
            level = rng.choice(["INFO", "DEBUG", "WARN"])
            msg = "".join(rng.choice("abcdefghijklmnop ") for _ in range(30))
            lines.append(f"[{level}] {msg}")
        new_content = "\n" + "\n".join(lines)

    return content + new_content.encode("utf-8")


def generate_local_edit(rng: random.Random, content: bytes, domain: str) -> bytes:
    """Generate local edit: small changes within the document."""
    content_str = content.decode("utf-8", errors="replace")

    # Pick 1-3 random positions for small edits
    num_edits = rng.randint(1, 3)
    positions = sorted(rng.sample(range(max(1, len(content_str) - 50)), min(num_edits, len(content_str) - 50)))

    result = content_str
    offset = 0

    for pos in positions:
        actual_pos = pos + offset
        if actual_pos >= len(result) - 10:
            continue

        # Replace 5-20 characters with different content
        edit_len = rng.randint(5, min(20, len(result) - actual_pos - 1))
        replacement = "".join(rng.choice("abcdefghij") for _ in range(edit_len))

        result = result[:actual_pos] + replacement + result[actual_pos + edit_len:]
        offset += len(replacement) - edit_len

    return result.encode("utf-8")


def generate_boundary_stress(
    rng: random.Random,
    content: bytes,
    boundaries: list[int],
    domain: str
) -> bytes:
    """Generate update near known chunk boundaries."""
    if not boundaries:
        return generate_local_edit(rng, content, domain)

    content_str = content.decode("utf-8", errors="replace")

    # Pick a boundary and edit near it
    boundary = rng.choice(boundaries)
    if boundary >= len(content_str):
        boundary = len(content_str) // 2

    # Edit within ±50 bytes of the boundary
    edit_start = max(0, boundary - rng.randint(0, 50))
    edit_end = min(len(content_str), edit_start + rng.randint(10, 30))

    replacement = "[[EDIT]]" + "".join(rng.choice("xyz123") for _ in range(rng.randint(5, 15)))
    result = content_str[:edit_start] + replacement + content_str[edit_end:]

    return result.encode("utf-8")


def generate_temporal_corpus(
    seed: int,
    num_docs: int,
    num_versions: int = 3,
) -> list[TemporalDocument]:
    """Generate a corpus with temporal versions (t0, t1, t2)."""
    rng = random.Random(seed)

    # Generate base corpus (t0)
    base_corpus = generate_corpus(
        seed=seed,
        num_docs=num_docs,
        domains=["text", "code", "json", "logs"],
        size_range=(2048, 8192),
    )

    temporal_docs = []
    scenarios = ["append", "edit", "boundary"]

    for doc in base_corpus:
        tdoc = TemporalDocument(
            doc_id=doc.doc_id,
            domain=doc.domain,
            versions=[doc.content],
            version_hashes=[hashlib.sha256(doc.content).hexdigest()],
            boundary_offsets=[doc.boundary_offsets],
            update_scenarios=[],
        )

        current_content = doc.content
        current_boundaries = doc.boundary_offsets

        # Generate subsequent versions
        for v in range(1, num_versions):
            # Rotate through scenarios for variety
            scenario = scenarios[(hash(doc.doc_id) + v) % len(scenarios)]
            tdoc.update_scenarios.append(scenario)

            if scenario == "append":
                new_content = generate_append_update(rng, current_content, doc.domain)
            elif scenario == "edit":
                new_content = generate_local_edit(rng, current_content, doc.domain)
            else:  # boundary
                new_content = generate_boundary_stress(rng, current_content, current_boundaries, doc.domain)

            tdoc.versions.append(new_content)
            tdoc.version_hashes.append(hashlib.sha256(new_content).hexdigest())

            # Update boundaries for next iteration (simplified - just add offset for append)
            if scenario == "append":
                size_diff = len(new_content) - len(current_content)
                current_boundaries = current_boundaries + [len(new_content)]

            tdoc.boundary_offsets.append(current_boundaries)
            current_content = new_content

        temporal_docs.append(tdoc)

    return temporal_docs


def baseline_chunk(data: bytes, config: Config) -> list[dict]:
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
        })

        offset += step
        idx += 1

        if offset >= len(data) and end == len(data):
            break

    return chunks


def stability_chunk(data: bytes, config: Config) -> list[dict]:
    """Stability-driven chunking."""
    chunk_objs = chunk_offline(data, config.chunking)
    return [
        {
            "chunk_index": i,
            "byte_offset_start": c.byte_start,
            "byte_offset_end": c.byte_end,
            "content": c.content,
            "content_sha256": hashlib.sha256(c.content).hexdigest(),
            "cut_score": c.cut_score,
        }
        for i, c in enumerate(chunk_objs)
    ]


def is_near_boundary(
    chunk_start: int,
    chunk_end: int,
    boundaries: list[int],
    window_bytes: int = 256,
) -> bool:
    """Check if chunk is near any known semantic boundary."""
    for b in boundaries:
        if abs(chunk_start - b) <= window_bytes or abs(chunk_end - b) <= window_bytes:
            return True
    return False


def compute_temporal_drift(
    embeddings_t0: dict[str, np.ndarray],
    embeddings_t1: dict[str, np.ndarray],
) -> DriftResult | None:
    """Compute drift between two time steps for content-stable chunks."""
    common = set(embeddings_t0.keys()) & set(embeddings_t1.keys())
    if len(common) < 2:
        return None

    old_emb = {k: embeddings_t0[k] for k in common}
    new_emb = {k: embeddings_t1[k] for k in common}

    return compute_drift_stats(old_emb, new_emb)


def run_temporal_experiment(
    run_name: str,
    method: str,  # "baseline" or "stability"
    config: Config,
    num_docs: int = 30,
    num_versions: int = 3,
) -> dict:
    """Run temporal stability experiment."""
    logger = logging.getLogger(__name__)

    git_version = get_git_version()
    output_dir = Path(config.general.output_dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting temporal experiment: {run_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Git version: tag={git_version.get('tag')}, commit={git_version.get('commit')}")

    # Generate temporal corpus
    logger.info(f"Generating temporal corpus: {num_docs} docs, {num_versions} versions")
    temporal_corpus = generate_temporal_corpus(
        seed=config.general.seed,
        num_docs=num_docs,
        num_versions=num_versions,
    )

    # Initialize embedding model
    logger.info("Loading embedding model")
    embed_model = EmbeddingModel(config.embedding)
    model_info = embed_model.model_info()

    # Process each time step
    all_versions_data = []  # List of dicts, one per time step

    for t in range(num_versions):
        logger.info(f"Processing time step t{t}")

        version_data = {
            "time_step": t,
            "chunks": [],
            "embeddings": {},  # content_sha256 -> embedding
            "chunk_boundaries": [],  # (doc_id, [boundary_offsets])
        }

        for tdoc in temporal_corpus:
            content = tdoc.versions[t]
            doc_boundaries = tdoc.boundary_offsets[t] if t < len(tdoc.boundary_offsets) else []

            # Chunk using selected method
            if method == "baseline":
                chunks = baseline_chunk(content, config)
            else:
                chunks = stability_chunk(content, config)

            # Embed and record chunks
            for chunk_data in chunks:
                chunk_id = f"{tdoc.doc_id}_t{t}_chunk_{chunk_data['chunk_index']}"

                # Check if near boundary
                near_boundary = is_near_boundary(
                    chunk_data["byte_offset_start"],
                    chunk_data["byte_offset_end"],
                    doc_boundaries,
                    window_bytes=config.eval.boundary_window_bytes,
                )

                # Embed
                text = chunk_data["content"].decode("utf-8", errors="replace")
                embedding = embed_model.embed_single(text)

                chunk_version = ChunkVersion(
                    chunk_id=chunk_id,
                    doc_id=tdoc.doc_id,
                    time_step=t,
                    byte_start=chunk_data["byte_offset_start"],
                    byte_end=chunk_data["byte_offset_end"],
                    content=chunk_data["content"],
                    content_sha256=chunk_data["content_sha256"],
                    embedding=embedding,
                    cut_score=chunk_data.get("cut_score", 0.0),
                    near_boundary=near_boundary,
                )

                version_data["chunks"].append(chunk_version)
                version_data["embeddings"][chunk_data["content_sha256"]] = embedding

            version_data["chunk_boundaries"].append((tdoc.doc_id, doc_boundaries))

        all_versions_data.append(version_data)
        logger.info(f"  t{t}: {len(version_data['chunks'])} chunks, {len(version_data['embeddings'])} unique")

    # Compute metrics across time steps
    logger.info("Computing temporal metrics")

    temporal_results = []
    all_drift_values = []
    all_churn_values = []
    boundary_drifts = []
    interior_drifts = []

    for t in range(1, num_versions):
        prev_data = all_versions_data[t - 1]
        curr_data = all_versions_data[t]

        # Drift for content-stable chunks
        drift_result = compute_temporal_drift(
            prev_data["embeddings"],
            curr_data["embeddings"],
        )

        # Maintenance cost
        prev_hashes = set(prev_data["embeddings"].keys())
        curr_hashes = set(curr_data["embeddings"].keys())
        maintenance = compute_maintenance_stats(prev_hashes, curr_hashes)

        # Churn: compare neighbor lists using probe embeddings
        # Use chunks that exist in both versions as probes
        common_hashes = prev_hashes & curr_hashes
        if len(common_hashes) >= 10:
            # Build simple neighbor lookup for each version
            prev_emb_list = list(prev_data["embeddings"].values())
            prev_hash_list = list(prev_data["embeddings"].keys())
            curr_emb_list = list(curr_data["embeddings"].values())
            curr_hash_list = list(curr_data["embeddings"].keys())

            prev_emb_matrix = np.array(prev_emb_list)
            curr_emb_matrix = np.array(curr_emb_list)

            # Sample probes from common chunks
            probe_hashes = list(common_hashes)[:min(50, len(common_hashes))]

            old_neighbors = []
            new_neighbors = []
            k = min(10, len(prev_hash_list) - 1, len(curr_hash_list) - 1)

            if k > 0:
                for probe_hash in probe_hashes:
                    probe_emb = prev_data["embeddings"][probe_hash]

                    # Find neighbors in prev version
                    prev_dists = np.linalg.norm(prev_emb_matrix - probe_emb, axis=1)
                    prev_indices = np.argsort(prev_dists)[1:k+1]  # Exclude self
                    prev_neighbors = [prev_hash_list[i] for i in prev_indices]

                    # Find neighbors in curr version
                    curr_dists = np.linalg.norm(curr_emb_matrix - probe_emb, axis=1)
                    curr_indices = np.argsort(curr_dists)[1:k+1]
                    curr_neighbors = [curr_hash_list[i] for i in curr_indices]

                    old_neighbors.append(prev_neighbors)
                    new_neighbors.append(curr_neighbors)

                churn_result = compute_churn_stats(old_neighbors, new_neighbors, k)
            else:
                churn_result = None
        else:
            churn_result = None

        # Boundary-localized drift
        for chunk in curr_data["chunks"]:
            if chunk.content_sha256 in prev_data["embeddings"]:
                prev_emb = prev_data["embeddings"][chunk.content_sha256]
                drift_val = float(np.linalg.norm(prev_emb - chunk.embedding))

                if chunk.near_boundary:
                    boundary_drifts.append(drift_val)
                else:
                    interior_drifts.append(drift_val)

        step_result = {
            "transition": f"t{t-1}->t{t}",
            "drift": drift_result.to_dict() if drift_result else None,
            "maintenance": maintenance.to_dict(),
            "churn": churn_result.to_dict() if churn_result else None,
        }

        if drift_result:
            all_drift_values.extend(drift_result.l2_distribution.tolist())
        if churn_result:
            all_churn_values.extend(churn_result.overlap_distribution.tolist())

        temporal_results.append(step_result)

        drift_str = f"{drift_result.mean_l2:.4f}" if drift_result else "N/A"
        logger.info(f"  {step_result['transition']}: "
                   f"drift_l2={drift_str}, "
                   f"reembed={maintenance.reembed_fraction:.2%}")

    # Aggregate metrics
    aggregate_metrics = TemporalMetrics()

    if all_drift_values:
        drift_arr = np.array(all_drift_values)
        aggregate_metrics.drift_l2_mean = float(np.mean(drift_arr))
        aggregate_metrics.drift_l2_p90 = float(np.percentile(drift_arr, 90))

    if all_churn_values:
        churn_arr = np.array(all_churn_values)
        aggregate_metrics.churn_topk_mean = float(np.mean(churn_arr))

    if boundary_drifts:
        aggregate_metrics.drift_near_boundary_mean = float(np.mean(boundary_drifts))
    if interior_drifts:
        aggregate_metrics.drift_interior_mean = float(np.mean(interior_drifts))

    # Sum maintenance across all transitions
    for tr in temporal_results:
        if tr["maintenance"]:
            aggregate_metrics.total_added += tr["maintenance"]["added_chunks"]
            aggregate_metrics.total_removed += tr["maintenance"]["removed_chunks"]
            aggregate_metrics.total_unchanged += tr["maintenance"]["unchanged_chunks"]

    total_chunks = aggregate_metrics.total_added + aggregate_metrics.total_unchanged
    if total_chunks > 0:
        aggregate_metrics.reembed_fraction = aggregate_metrics.total_added / total_chunks

    # Save results
    summary = {
        "run_name": run_name,
        "method": method,
        "git_version": git_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "corpus": {
            "num_docs": num_docs,
            "num_versions": num_versions,
            "update_scenarios": ["append", "edit", "boundary"],
        },
        "aggregate_metrics": aggregate_metrics.to_dict(),
        "temporal_transitions": temporal_results,
        "per_version_stats": [
            {
                "time_step": v["time_step"],
                "num_chunks": len(v["chunks"]),
                "num_unique_content": len(v["embeddings"]),
            }
            for v in all_versions_data
        ],
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save boundary-localized analysis
    boundary_analysis = {
        "boundary_window_bytes": config.eval.boundary_window_bytes,
        "commit_horizon_bytes": config.chunking.commit_horizon_bytes,
        "boundary_drift_values": boundary_drifts,
        "interior_drift_values": interior_drifts,
        "boundary_drift_mean": aggregate_metrics.drift_near_boundary_mean,
        "interior_drift_mean": aggregate_metrics.drift_interior_mean,
    }
    with open(run_dir / "boundary_analysis.json", "w") as f:
        json.dump(boundary_analysis, f, indent=2)

    logger.info(f"Experiment complete. Results saved to {run_dir}")
    return summary


def generate_comparison_plots(
    baseline_summary: dict,
    stability_summary: dict,
    output_dir: Path,
    date_suffix: str,
) -> None:
    """Generate paper-quality comparison plots."""
    logger = logging.getLogger(__name__)
    logger.info("Generating comparison plots")

    figures_dir = output_dir.parent / "paper" / "figures"
    tables_dir = output_dir.parent / "paper" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Set paper style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.figsize": (6, 4),
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # Get commit horizon for captions
    commit_horizon = baseline_summary["config"]["chunking"]["commit_horizon_bytes"]

    # 1. Drift comparison plot
    fig, ax = plt.subplots()

    methods = ["Baseline", "Stability"]
    drift_means = [
        baseline_summary["aggregate_metrics"]["drift_l2_mean"],
        stability_summary["aggregate_metrics"]["drift_l2_mean"],
    ]
    drift_p90s = [
        baseline_summary["aggregate_metrics"]["drift_l2_p90"],
        stability_summary["aggregate_metrics"]["drift_l2_p90"],
    ]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, drift_means, width, label="Mean", color="#1f77b4")
    bars2 = ax.bar(x + width/2, drift_p90s, width, label="P90", color="#ff7f0e")

    ax.set_ylabel("Embedding Drift (L2)")
    ax.set_title(f"Embedding Drift Under Updates\n(commit_horizon={commit_horizon} bytes)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    fig.savefig(figures_dir / f"temporal_drift_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Churn comparison plot
    fig, ax = plt.subplots()

    churn_values = [
        baseline_summary["aggregate_metrics"]["churn_topk_mean"],
        stability_summary["aggregate_metrics"]["churn_topk_mean"],
    ]

    bars = ax.bar(methods, churn_values, color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("Top-k Overlap")
    ax.set_title(f"ANN Neighbor Stability Under Updates\n(commit_horizon={commit_horizon} bytes)")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, churn_values):
        ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.savefig(figures_dir / f"temporal_churn_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Maintenance cost plot
    fig, ax = plt.subplots()

    reembed_fractions = [
        baseline_summary["aggregate_metrics"]["reembed_fraction"],
        stability_summary["aggregate_metrics"]["reembed_fraction"],
    ]

    bars = ax.bar(methods, reembed_fractions, color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("Re-embed Fraction")
    ax.set_title(f"Maintenance Cost Under Updates\n(commit_horizon={commit_horizon} bytes)")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, reembed_fractions):
        ax.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.savefig(figures_dir / f"temporal_maintenance_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 4. Boundary-localized effects
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Drift by region
    ax1 = axes[0]
    regions = ["Near Boundary", "Interior"]
    baseline_drifts = [
        baseline_summary["aggregate_metrics"]["drift_near_boundary_mean"],
        baseline_summary["aggregate_metrics"]["drift_interior_mean"],
    ]
    stability_drifts = [
        stability_summary["aggregate_metrics"]["drift_near_boundary_mean"],
        stability_summary["aggregate_metrics"]["drift_interior_mean"],
    ]

    x = np.arange(len(regions))
    width = 0.35
    ax1.bar(x - width/2, baseline_drifts, width, label="Baseline", color="#1f77b4")
    ax1.bar(x + width/2, stability_drifts, width, label="Stability", color="#2ca02c")
    ax1.set_ylabel("Mean Drift (L2)")
    ax1.set_title("Drift by Region")
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions)
    ax1.legend()

    # Chunks over time
    ax2 = axes[1]
    baseline_chunks = [v["num_chunks"] for v in baseline_summary["per_version_stats"]]
    stability_chunks = [v["num_chunks"] for v in stability_summary["per_version_stats"]]
    time_steps = [f"t{v['time_step']}" for v in baseline_summary["per_version_stats"]]

    ax2.plot(time_steps, baseline_chunks, marker="o", label="Baseline", color="#1f77b4")
    ax2.plot(time_steps, stability_chunks, marker="s", label="Stability", color="#2ca02c")
    ax2.set_ylabel("Chunk Count")
    ax2.set_xlabel("Time Step")
    ax2.set_title("Chunk Count Over Time")
    ax2.legend()

    fig.suptitle(f"Boundary-Localized Analysis (commit_horizon={commit_horizon} bytes)", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / f"temporal_boundary_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Generate LaTeX table
    table_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Temporal Stability Metrics (commit\\_horizon={commit_horizon} bytes)}}
\\label{{tab:temporal-stability}}
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{Stability}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Embedding Drift}}}} \\\\
Mean L2 & {baseline_summary["aggregate_metrics"]["drift_l2_mean"]:.4f} & {stability_summary["aggregate_metrics"]["drift_l2_mean"]:.4f} \\\\
P90 L2 & {baseline_summary["aggregate_metrics"]["drift_l2_p90"]:.4f} & {stability_summary["aggregate_metrics"]["drift_l2_p90"]:.4f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{ANN Neighbor Churn}}}} \\\\
Top-k Overlap & {baseline_summary["aggregate_metrics"]["churn_topk_mean"]:.3f} & {stability_summary["aggregate_metrics"]["churn_topk_mean"]:.3f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Maintenance Cost}}}} \\\\
Re-embed Fraction & {baseline_summary["aggregate_metrics"]["reembed_fraction"]:.1%} & {stability_summary["aggregate_metrics"]["reembed_fraction"]:.1%} \\\\
Chunks Added & {baseline_summary["aggregate_metrics"]["total_added"]} & {stability_summary["aggregate_metrics"]["total_added"]} \\\\
Chunks Unchanged & {baseline_summary["aggregate_metrics"]["total_unchanged"]} & {stability_summary["aggregate_metrics"]["total_unchanged"]} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Boundary Effects}}}} \\\\
Drift (Near Boundary) & {baseline_summary["aggregate_metrics"]["drift_near_boundary_mean"]:.4f} & {stability_summary["aggregate_metrics"]["drift_near_boundary_mean"]:.4f} \\\\
Drift (Interior) & {baseline_summary["aggregate_metrics"]["drift_interior_mean"]:.4f} & {stability_summary["aggregate_metrics"]["drift_interior_mean"]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with open(tables_dir / f"temporal_stability_{date_suffix}.tex", "w") as f:
        f.write(table_content)

    logger.info(f"Saved plots to {figures_dir}")
    logger.info(f"Saved table to {tables_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run temporal stability experiment"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "stability"],
        required=True,
        help="Chunking method to use",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=30,
        help="Number of documents in corpus",
    )
    parser.add_argument(
        "--num-versions",
        type=int,
        default=3,
        help="Number of temporal versions (t0, t1, ...)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison plots (requires both baseline and stability runs)",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        help="Path to baseline run summary.json (for --compare)",
    )
    parser.add_argument(
        "--stability-run",
        type=str,
        help="Path to stability run summary.json (for --compare)",
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

    if args.compare:
        # Generate comparison plots from existing runs
        if not args.baseline_run or not args.stability_run:
            print("Error: --compare requires --baseline-run and --stability-run", file=sys.stderr)
            return 1

        with open(args.baseline_run) as f:
            baseline_summary = json.load(f)
        with open(args.stability_run) as f:
            stability_summary = json.load(f)

        output_dir = Path(config.general.output_dir)
        date_suffix = datetime.now().strftime("%Y%m%d")

        generate_comparison_plots(
            baseline_summary,
            stability_summary,
            output_dir,
            f"v1.3.0-{date_suffix}",
        )
        return 0

    try:
        summary = run_temporal_experiment(
            run_name=args.run_name,
            method=args.method,
            config=config,
            num_docs=args.num_docs,
            num_versions=args.num_versions,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
