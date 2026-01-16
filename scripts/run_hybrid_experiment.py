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
Hybrid chunking temporal experiment for curv-embedding v1.5.0.

Compares three chunking policies under incremental updates:
1. Baseline: Fixed-size overlapping chunks
2. Stability: Stability-driven chunks
3. Hybrid: Stability base + localized micro-chunks in edit windows

v1.4.0 Takeaway (incorporated):
Stability-driven chunking optimizes representational coherence but increases
sensitivity to localized edits due to larger chunk granularity, while fixed-size
overlapping chunking provides higher mutation tolerance by limiting invalidation
scope. Representational stability and mutation tolerance are distinct, competing
objectives.

Usage:
    uv run scripts/run_hybrid_experiment.py --run-name v1.4.0-baseline-temporal-20260116 --method baseline
    uv run scripts/run_hybrid_experiment.py --run-name v1.4.0-stability-temporal-20260116 --method stability
    uv run scripts/run_hybrid_experiment.py --run-name v1.4.0-hybrid-temporal-20260116 --method hybrid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config, HybridConfig
from src.data.generator import generate_corpus, SyntheticDocument
from src.chunking.offline import chunk_offline
from src.embedding.model import EmbeddingModel
from src.eval.drift import compute_drift_stats, DriftResult
from src.eval.churn import compute_churn_stats, ChurnResult
from src.eval.maintenance import compute_maintenance_stats, MaintenanceResult


# v1.4.0 Takeaway (mandatory incorporation)
V140_TAKEAWAY = """Stability-driven chunking optimizes representational coherence but increases
sensitivity to localized edits due to larger chunk granularity, while fixed-size
overlapping chunking provides higher mutation tolerance by limiting invalidation
scope. Representational stability and mutation tolerance are distinct, competing
objectives."""


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
class EditWindow:
    """Detected edit region in a document."""
    doc_id: str
    edit_start: int
    edit_end: int
    guarded_start: int
    guarded_end: int
    edit_type: str  # append, edit, boundary


@dataclass
class HybridChunk:
    """A chunk with hybrid chunking metadata."""
    chunk_id: str
    doc_id: str
    time_step: int
    byte_start: int
    byte_end: int
    content: bytes
    content_sha256: str
    embedding: np.ndarray | None = None
    cut_score: float = 0.0
    is_micro_chunk: bool = False
    parent_chunk_id: str | None = None
    edit_window_id: str | None = None
    near_boundary: bool = False


@dataclass
class TemporalDocument:
    """A document with temporal versions."""
    doc_id: str
    domain: str
    versions: list[bytes] = field(default_factory=list)
    version_hashes: list[str] = field(default_factory=list)
    boundary_offsets: list[list[int]] = field(default_factory=list)
    update_scenarios: list[str] = field(default_factory=list)
    edit_windows: list[EditWindow | None] = field(default_factory=list)


@dataclass
class HybridMetrics:
    """Extended temporal stability metrics for hybrid evaluation."""
    # Core metrics
    drift_l2_mean: float = 0.0
    drift_l2_p90: float = 0.0
    churn_topk_mean: float = 0.0
    reembed_fraction: float = 0.0
    total_added: int = 0
    total_removed: int = 0
    total_unchanged: int = 0

    # Structural metrics
    total_chunks: int = 0
    mean_chunk_size: float = 0.0
    p90_chunk_size: float = 0.0

    # Hybrid-specific metrics
    micro_chunk_fraction: float = 0.0
    base_chunk_fraction: float = 0.0
    reembeds_in_edit_window: int = 0
    reembeds_outside_edit_window: int = 0
    localization_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "drift_l2_mean": self.drift_l2_mean,
            "drift_l2_p90": self.drift_l2_p90,
            "churn_topk_mean": self.churn_topk_mean,
            "reembed_fraction": self.reembed_fraction,
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "total_unchanged": self.total_unchanged,
            "total_chunks": self.total_chunks,
            "mean_chunk_size": self.mean_chunk_size,
            "p90_chunk_size": self.p90_chunk_size,
            "micro_chunk_fraction": self.micro_chunk_fraction,
            "base_chunk_fraction": self.base_chunk_fraction,
            "reembeds_in_edit_window": self.reembeds_in_edit_window,
            "reembeds_outside_edit_window": self.reembeds_outside_edit_window,
            "localization_efficiency": self.localization_efficiency,
        }


def generate_append_update(rng: random.Random, content: bytes, domain: str) -> tuple[bytes, EditWindow]:
    """Generate append-only update with edit window tracking."""
    new_size = rng.randint(200, 800)
    edit_start = len(content)

    if domain == "text":
        words = []
        while len(" ".join(words).encode()) < new_size:
            word_len = rng.randint(3, 10)
            word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_len))
            words.append(word)
            if rng.random() < 0.1:
                words.append(".")
        new_content = "\n\n" + " ".join(words)
    elif domain == "code":
        func_name = "_".join(rng.choice(["get", "set", "update", "process", "handle"])
                            for _ in range(2))
        new_content = f"\n\ndef {func_name}(arg):\n    # New function\n    return arg\n"
    elif domain == "json":
        new_content = ',\n  {"new_field": "' + "".join(rng.choice("abcdef") for _ in range(20)) + '"}'
    else:
        lines = []
        for _ in range(rng.randint(5, 15)):
            level = rng.choice(["INFO", "DEBUG", "WARN"])
            msg = "".join(rng.choice("abcdefghijklmnop ") for _ in range(30))
            lines.append(f"[{level}] {msg}")
        new_content = "\n" + "\n".join(lines)

    result = content + new_content.encode("utf-8")
    edit_end = len(result)

    edit_window = EditWindow(
        doc_id="",  # filled later
        edit_start=edit_start,
        edit_end=edit_end,
        guarded_start=edit_start,  # guard band applied later
        guarded_end=edit_end,
        edit_type="append",
    )

    return result, edit_window


def generate_local_edit(rng: random.Random, content: bytes, domain: str) -> tuple[bytes, EditWindow]:
    """Generate local edit with edit window tracking."""
    content_str = content.decode("utf-8", errors="replace")

    # Pick a random position for the edit
    if len(content_str) < 50:
        pos = 0
    else:
        pos = rng.randint(0, max(1, len(content_str) - 50))

    edit_len = rng.randint(5, min(20, len(content_str) - pos - 1)) if pos < len(content_str) - 1 else 5
    replacement = "".join(rng.choice("abcdefghij") for _ in range(edit_len + rng.randint(-2, 5)))

    result = content_str[:pos] + replacement + content_str[pos + edit_len:]

    edit_window = EditWindow(
        doc_id="",
        edit_start=pos,
        edit_end=pos + len(replacement),
        guarded_start=pos,
        guarded_end=pos + len(replacement),
        edit_type="edit",
    )

    return result.encode("utf-8"), edit_window


def generate_boundary_stress(
    rng: random.Random,
    content: bytes,
    boundaries: list[int],
    domain: str
) -> tuple[bytes, EditWindow]:
    """Generate update near known chunk boundaries with edit window tracking."""
    if not boundaries:
        return generate_local_edit(rng, content, domain)

    content_str = content.decode("utf-8", errors="replace")

    boundary = rng.choice(boundaries)
    if boundary >= len(content_str):
        boundary = len(content_str) // 2

    edit_start = max(0, boundary - rng.randint(0, 50))
    edit_end = min(len(content_str), edit_start + rng.randint(10, 30))

    replacement = "[[EDIT]]" + "".join(rng.choice("xyz123") for _ in range(rng.randint(5, 15)))
    result = content_str[:edit_start] + replacement + content_str[edit_end:]

    edit_window = EditWindow(
        doc_id="",
        edit_start=edit_start,
        edit_end=edit_start + len(replacement),
        guarded_start=edit_start,
        guarded_end=edit_start + len(replacement),
        edit_type="boundary",
    )

    return result.encode("utf-8"), edit_window


def generate_temporal_corpus(
    seed: int,
    num_docs: int,
    num_versions: int = 3,
    guard_band_bytes: int = 256,
) -> list[TemporalDocument]:
    """Generate a corpus with temporal versions and edit window tracking."""
    rng = random.Random(seed)

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
            edit_windows=[None],  # No edit at t0
        )

        current_content = doc.content
        current_boundaries = doc.boundary_offsets

        for v in range(1, num_versions):
            scenario = scenarios[(hash(doc.doc_id) + v) % len(scenarios)]
            tdoc.update_scenarios.append(scenario)

            if scenario == "append":
                new_content, edit_window = generate_append_update(rng, current_content, doc.domain)
            elif scenario == "edit":
                new_content, edit_window = generate_local_edit(rng, current_content, doc.domain)
            else:
                new_content, edit_window = generate_boundary_stress(rng, current_content, current_boundaries, doc.domain)

            # Apply guard band
            edit_window.doc_id = doc.doc_id
            edit_window.guarded_start = max(0, edit_window.edit_start - guard_band_bytes)
            edit_window.guarded_end = min(len(new_content), edit_window.edit_end + guard_band_bytes)

            tdoc.versions.append(new_content)
            tdoc.version_hashes.append(hashlib.sha256(new_content).hexdigest())
            tdoc.edit_windows.append(edit_window)

            if scenario == "append":
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
            "is_micro_chunk": False,
            "parent_chunk_id": None,
            "edit_window_id": None,
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
            "is_micro_chunk": False,
            "parent_chunk_id": None,
            "edit_window_id": None,
        }
        for i, c in enumerate(chunk_objs)
    ]


def micro_chunk(
    data: bytes,
    start: int,
    end: int,
    config: HybridConfig,
    parent_chunk_id: str,
    edit_window_id: str,
) -> list[dict]:
    """Create micro-chunks within an edit window."""
    window_data = data[start:end]
    chunk_size = config.micro_chunk_bytes
    overlap = config.micro_overlap_bytes
    step = chunk_size - overlap

    chunks = []
    offset = 0
    idx = 0

    while offset < len(window_data):
        chunk_end = min(offset + chunk_size, len(window_data))
        content = window_data[offset:chunk_end]
        content_hash = hashlib.sha256(content).hexdigest()

        chunks.append({
            "chunk_index": idx,
            "byte_offset_start": start + offset,
            "byte_offset_end": start + chunk_end,
            "content": content,
            "content_sha256": content_hash,
            "cut_score": 0.0,
            "is_micro_chunk": True,
            "parent_chunk_id": parent_chunk_id,
            "edit_window_id": edit_window_id,
        })

        offset += step
        idx += 1

        if offset >= len(window_data) and chunk_end == len(window_data):
            break

    return chunks


def hybrid_chunk(
    data: bytes,
    prev_chunks: list[dict] | None,
    edit_window: EditWindow | None,
    config: Config,
) -> list[dict]:
    """Hybrid chunking: stability base + micro-chunks in edit windows.

    At t0 (no prev_chunks): perform full stability-driven chunking.
    At t1+: keep base chunks outside edit window, micro-chunk inside.
    """
    if prev_chunks is None or edit_window is None:
        # Initial chunking - use stability-driven
        return stability_chunk(data, config)

    # Identify chunks affected by the edit window
    guarded_start = edit_window.guarded_start
    guarded_end = edit_window.guarded_end
    edit_window_id = str(uuid.uuid4())[:8]

    result_chunks = []
    micro_chunk_parent = None

    for chunk in prev_chunks:
        chunk_start = chunk["byte_offset_start"]
        chunk_end = chunk["byte_offset_end"]

        # Check if chunk overlaps with guarded edit window
        overlaps = not (chunk_end <= guarded_start or chunk_start >= guarded_end)

        if overlaps:
            # This chunk is affected - record as parent for micro-chunks
            if micro_chunk_parent is None:
                micro_chunk_parent = chunk["content_sha256"]
        else:
            # Chunk is outside edit window - keep it if content unchanged
            # For simplicity, we re-extract at same offsets (in real use, would verify content)
            if chunk_end <= len(data):
                new_content = data[chunk_start:chunk_end]
                new_hash = hashlib.sha256(new_content).hexdigest()

                result_chunks.append({
                    "chunk_index": len(result_chunks),
                    "byte_offset_start": chunk_start,
                    "byte_offset_end": chunk_end,
                    "content": new_content,
                    "content_sha256": new_hash,
                    "cut_score": chunk.get("cut_score", 0.0),
                    "is_micro_chunk": False,
                    "parent_chunk_id": None,
                    "edit_window_id": None,
                })

    # Create micro-chunks for the edit window
    if guarded_end <= len(data):
        micro_chunks = micro_chunk(
            data,
            guarded_start,
            min(guarded_end, len(data)),
            config.hybrid,
            micro_chunk_parent or "unknown",
            edit_window_id,
        )

        # Update indices for micro-chunks
        for mc in micro_chunks:
            mc["chunk_index"] = len(result_chunks)
            result_chunks.append(mc)

    # Sort by byte offset
    result_chunks.sort(key=lambda c: c["byte_offset_start"])

    # Re-index
    for i, c in enumerate(result_chunks):
        c["chunk_index"] = i

    return result_chunks


def is_in_edit_window(chunk_start: int, chunk_end: int, edit_window: EditWindow | None) -> bool:
    """Check if chunk overlaps with edit window."""
    if edit_window is None:
        return False
    return not (chunk_end <= edit_window.guarded_start or chunk_start >= edit_window.guarded_end)


def run_hybrid_experiment(
    run_name: str,
    method: str,  # "baseline", "stability", or "hybrid"
    config: Config,
    num_docs: int = 30,
    num_versions: int = 3,
) -> dict:
    """Run hybrid temporal stability experiment."""
    logger = logging.getLogger(__name__)

    git_version = get_git_version()
    output_dir = Path(config.general.output_dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting hybrid experiment: {run_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Git version: tag={git_version.get('tag')}, commit={git_version.get('commit')}")

    # Generate temporal corpus with edit window tracking
    logger.info(f"Generating temporal corpus: {num_docs} docs, {num_versions} versions")
    temporal_corpus = generate_temporal_corpus(
        seed=config.general.seed,
        num_docs=num_docs,
        num_versions=num_versions,
        guard_band_bytes=config.hybrid.guard_band_bytes,
    )

    # Initialize embedding model
    logger.info("Loading embedding model")
    embed_model = EmbeddingModel(config.embedding)

    # Track chunks per document for hybrid chunking
    doc_prev_chunks: dict[str, list[dict]] = {}

    # Process each time step
    all_versions_data = []

    for t in range(num_versions):
        logger.info(f"Processing time step t{t}")

        version_data = {
            "time_step": t,
            "chunks": [],
            "embeddings": {},
            "chunk_sizes": [],
            "micro_chunk_count": 0,
            "base_chunk_count": 0,
        }

        for tdoc in temporal_corpus:
            content = tdoc.versions[t]
            edit_window = tdoc.edit_windows[t] if t < len(tdoc.edit_windows) else None
            prev_chunks = doc_prev_chunks.get(tdoc.doc_id)

            # Chunk using selected method
            if method == "baseline":
                chunks = baseline_chunk(content, config)
            elif method == "stability":
                chunks = stability_chunk(content, config)
            else:  # hybrid
                chunks = hybrid_chunk(content, prev_chunks, edit_window, config)

            # Store for next iteration
            doc_prev_chunks[tdoc.doc_id] = chunks

            # Embed and record chunks
            for chunk_data in chunks:
                chunk_id = f"{tdoc.doc_id}_t{t}_chunk_{chunk_data['chunk_index']}"

                # Embed
                text = chunk_data["content"].decode("utf-8", errors="replace")
                embedding = embed_model.embed_single(text)

                chunk = HybridChunk(
                    chunk_id=chunk_id,
                    doc_id=tdoc.doc_id,
                    time_step=t,
                    byte_start=chunk_data["byte_offset_start"],
                    byte_end=chunk_data["byte_offset_end"],
                    content=chunk_data["content"],
                    content_sha256=chunk_data["content_sha256"],
                    embedding=embedding,
                    cut_score=chunk_data.get("cut_score", 0.0),
                    is_micro_chunk=chunk_data.get("is_micro_chunk", False),
                    parent_chunk_id=chunk_data.get("parent_chunk_id"),
                    edit_window_id=chunk_data.get("edit_window_id"),
                    near_boundary=is_in_edit_window(
                        chunk_data["byte_offset_start"],
                        chunk_data["byte_offset_end"],
                        edit_window,
                    ),
                )

                version_data["chunks"].append(chunk)
                version_data["embeddings"][chunk_data["content_sha256"]] = embedding
                version_data["chunk_sizes"].append(chunk_data["byte_offset_end"] - chunk_data["byte_offset_start"])

                if chunk_data.get("is_micro_chunk", False):
                    version_data["micro_chunk_count"] += 1
                else:
                    version_data["base_chunk_count"] += 1

        all_versions_data.append(version_data)
        total_chunks = len(version_data["chunks"])
        micro_frac = version_data["micro_chunk_count"] / total_chunks if total_chunks > 0 else 0
        logger.info(f"  t{t}: {total_chunks} chunks ({version_data['micro_chunk_count']} micro, {version_data['base_chunk_count']} base, {micro_frac:.1%} micro)")

    # Compute metrics across time steps
    logger.info("Computing temporal metrics")

    temporal_results = []
    all_drift_values = []
    all_churn_values = []

    for t in range(1, num_versions):
        prev_data = all_versions_data[t - 1]
        curr_data = all_versions_data[t]

        # Drift for content-stable chunks
        common = set(prev_data["embeddings"].keys()) & set(curr_data["embeddings"].keys())
        if len(common) >= 2:
            old_emb = {k: prev_data["embeddings"][k] for k in common}
            new_emb = {k: curr_data["embeddings"][k] for k in common}
            drift_result = compute_drift_stats(old_emb, new_emb)
        else:
            drift_result = None

        # Maintenance cost
        prev_hashes = set(prev_data["embeddings"].keys())
        curr_hashes = set(curr_data["embeddings"].keys())
        maintenance = compute_maintenance_stats(prev_hashes, curr_hashes)

        # Count re-embeds in/outside edit windows
        added_hashes = curr_hashes - prev_hashes
        reembeds_in_window = 0
        reembeds_outside_window = 0

        for chunk in curr_data["chunks"]:
            if chunk.content_sha256 in added_hashes:
                if chunk.near_boundary:
                    reembeds_in_window += 1
                else:
                    reembeds_outside_window += 1

        # Churn computation
        if len(common) >= 10:
            prev_emb_list = list(prev_data["embeddings"].values())
            prev_hash_list = list(prev_data["embeddings"].keys())
            curr_emb_list = list(curr_data["embeddings"].values())
            curr_hash_list = list(curr_data["embeddings"].keys())

            prev_emb_matrix = np.array(prev_emb_list)
            curr_emb_matrix = np.array(curr_emb_list)

            probe_hashes = list(common)[:min(50, len(common))]
            old_neighbors = []
            new_neighbors = []
            k = min(10, len(prev_hash_list) - 1, len(curr_hash_list) - 1)

            if k > 0:
                for probe_hash in probe_hashes:
                    probe_emb = prev_data["embeddings"][probe_hash]

                    prev_dists = np.linalg.norm(prev_emb_matrix - probe_emb, axis=1)
                    prev_indices = np.argsort(prev_dists)[1:k+1]
                    prev_neighbors = [prev_hash_list[i] for i in prev_indices]

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

        step_result = {
            "transition": f"t{t-1}->t{t}",
            "drift": drift_result.to_dict() if drift_result else None,
            "maintenance": maintenance.to_dict(),
            "churn": churn_result.to_dict() if churn_result else None,
            "reembeds_in_window": reembeds_in_window,
            "reembeds_outside_window": reembeds_outside_window,
        }

        if drift_result:
            all_drift_values.extend(drift_result.l2_distribution.tolist())
        if churn_result:
            all_churn_values.extend(churn_result.overlap_distribution.tolist())

        temporal_results.append(step_result)

        drift_str = f"{drift_result.mean_l2:.4f}" if drift_result else "N/A"
        logger.info(f"  {step_result['transition']}: "
                   f"drift_l2={drift_str}, "
                   f"reembed={maintenance.reembed_fraction:.2%}, "
                   f"in_window={reembeds_in_window}, out={reembeds_outside_window}")

    # Aggregate metrics
    aggregate_metrics = HybridMetrics()

    if all_drift_values:
        drift_arr = np.array(all_drift_values)
        aggregate_metrics.drift_l2_mean = float(np.mean(drift_arr))
        aggregate_metrics.drift_l2_p90 = float(np.percentile(drift_arr, 90))

    if all_churn_values:
        churn_arr = np.array(all_churn_values)
        aggregate_metrics.churn_topk_mean = float(np.mean(churn_arr))

    for tr in temporal_results:
        if tr["maintenance"]:
            aggregate_metrics.total_added += tr["maintenance"]["added_chunks"]
            aggregate_metrics.total_removed += tr["maintenance"]["removed_chunks"]
            aggregate_metrics.total_unchanged += tr["maintenance"]["unchanged_chunks"]
        aggregate_metrics.reembeds_in_edit_window += tr["reembeds_in_window"]
        aggregate_metrics.reembeds_outside_edit_window += tr["reembeds_outside_window"]

    total_chunks = aggregate_metrics.total_added + aggregate_metrics.total_unchanged
    if total_chunks > 0:
        aggregate_metrics.reembed_fraction = aggregate_metrics.total_added / total_chunks

    # Localization efficiency
    total_reembeds = aggregate_metrics.reembeds_in_edit_window + aggregate_metrics.reembeds_outside_edit_window
    if total_reembeds > 0:
        aggregate_metrics.localization_efficiency = aggregate_metrics.reembeds_in_edit_window / total_reembeds

    # Final version stats
    final_data = all_versions_data[-1]
    aggregate_metrics.total_chunks = len(final_data["chunks"])
    if final_data["chunk_sizes"]:
        aggregate_metrics.mean_chunk_size = float(np.mean(final_data["chunk_sizes"]))
        aggregate_metrics.p90_chunk_size = float(np.percentile(final_data["chunk_sizes"], 90))

    total_final = final_data["micro_chunk_count"] + final_data["base_chunk_count"]
    if total_final > 0:
        aggregate_metrics.micro_chunk_fraction = final_data["micro_chunk_count"] / total_final
        aggregate_metrics.base_chunk_fraction = final_data["base_chunk_count"] / total_final

    # Save results
    summary = {
        "run_name": run_name,
        "method": method,
        "git_version": git_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "v140_takeaway": V140_TAKEAWAY,
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
                "micro_chunks": v["micro_chunk_count"],
                "base_chunks": v["base_chunk_count"],
                "mean_chunk_size": float(np.mean(v["chunk_sizes"])) if v["chunk_sizes"] else 0,
            }
            for v in all_versions_data
        ],
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment complete. Results saved to {run_dir}")
    return summary


def generate_three_way_comparison(
    baseline_summary: dict,
    stability_summary: dict,
    hybrid_summary: dict,
    output_dir: Path,
    date_suffix: str,
) -> None:
    """Generate paper-quality three-way comparison plots."""
    logger = logging.getLogger(__name__)
    logger.info("Generating three-way comparison plots")

    figures_dir = output_dir.parent / "paper" / "figures"
    tables_dir = output_dir.parent / "paper" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.figsize": (8, 5),
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    methods = ["Baseline", "Stability", "Hybrid"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    commit_horizon = baseline_summary["config"]["chunking"]["commit_horizon_bytes"]

    # 1. Main comparison: Re-embed fraction + Churn + Chunks
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Re-embed fraction
    ax1 = axes[0]
    reembed_values = [
        baseline_summary["aggregate_metrics"]["reembed_fraction"],
        stability_summary["aggregate_metrics"]["reembed_fraction"],
        hybrid_summary["aggregate_metrics"]["reembed_fraction"],
    ]
    bars = ax1.bar(methods, reembed_values, color=colors)
    ax1.set_ylabel("Re-embed Fraction")
    ax1.set_title("Mutation Tolerance")
    ax1.set_ylim(0, 1.0)
    for bar, val in zip(bars, reembed_values):
        ax1.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Churn (top-k overlap - higher is better)
    ax2 = axes[1]
    churn_values = [
        baseline_summary["aggregate_metrics"]["churn_topk_mean"],
        stability_summary["aggregate_metrics"]["churn_topk_mean"],
        hybrid_summary["aggregate_metrics"]["churn_topk_mean"],
    ]
    bars = ax2.bar(methods, churn_values, color=colors)
    ax2.set_ylabel("Top-k Overlap")
    ax2.set_title("Neighbor Stability")
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, churn_values):
        ax2.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Total chunks (lower is better for efficiency)
    ax3 = axes[2]
    chunk_values = [
        baseline_summary["aggregate_metrics"]["total_chunks"],
        stability_summary["aggregate_metrics"]["total_chunks"],
        hybrid_summary["aggregate_metrics"]["total_chunks"],
    ]
    bars = ax3.bar(methods, chunk_values, color=colors)
    ax3.set_ylabel("Chunk Count")
    ax3.set_title("Structural Overhead")
    for bar, val in zip(bars, chunk_values):
        ax3.annotate(f"{val}", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle(f"Three-Way Chunking Comparison (commit_horizon={commit_horizon} bytes)", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / f"temporal_hybrid_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Localization efficiency (hybrid-specific)
    fig, ax = plt.subplots(figsize=(6, 4))

    loc_eff = [
        0,  # Baseline doesn't have edit window concept
        0,  # Stability doesn't have edit window concept
        hybrid_summary["aggregate_metrics"]["localization_efficiency"],
    ]
    bars = ax.bar(methods, loc_eff, color=colors)
    ax.set_ylabel("Localization Efficiency")
    ax.set_title("Re-embeds Confined to Edit Windows")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, loc_eff):
        ax.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.savefig(figures_dir / f"temporal_localization_{date_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Generate LaTeX table
    table_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Hybrid Chunking Comparison (commit\\_horizon={commit_horizon} bytes)}}
\\label{{tab:hybrid-comparison}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{Stability}} & \\textbf{{Hybrid}} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Mutation Tolerance}}}} \\\\
Re-embed Fraction & {baseline_summary["aggregate_metrics"]["reembed_fraction"]:.1%} & {stability_summary["aggregate_metrics"]["reembed_fraction"]:.1%} & {hybrid_summary["aggregate_metrics"]["reembed_fraction"]:.1%} \\\\
Chunks Unchanged & {baseline_summary["aggregate_metrics"]["total_unchanged"]} & {stability_summary["aggregate_metrics"]["total_unchanged"]} & {hybrid_summary["aggregate_metrics"]["total_unchanged"]} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Neighbor Churn}}}} \\\\
Top-k Overlap & {baseline_summary["aggregate_metrics"]["churn_topk_mean"]:.3f} & {stability_summary["aggregate_metrics"]["churn_topk_mean"]:.3f} & {hybrid_summary["aggregate_metrics"]["churn_topk_mean"]:.3f} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Structural Overhead}}}} \\\\
Total Chunks & {baseline_summary["aggregate_metrics"]["total_chunks"]} & {stability_summary["aggregate_metrics"]["total_chunks"]} & {hybrid_summary["aggregate_metrics"]["total_chunks"]} \\\\
Mean Chunk Size & {baseline_summary["aggregate_metrics"]["mean_chunk_size"]:.0f} & {stability_summary["aggregate_metrics"]["mean_chunk_size"]:.0f} & {hybrid_summary["aggregate_metrics"]["mean_chunk_size"]:.0f} \\\\
Micro-chunk Fraction & -- & -- & {hybrid_summary["aggregate_metrics"]["micro_chunk_fraction"]:.1%} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Localization (Hybrid only)}}}} \\\\
Localization Efficiency & -- & -- & {hybrid_summary["aggregate_metrics"]["localization_efficiency"]:.1%} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with open(tables_dir / f"temporal_hybrid_{date_suffix}.tex", "w") as f:
        f.write(table_content)

    logger.info(f"Saved plots to {figures_dir}")
    logger.info(f"Saved table to {tables_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run hybrid temporal stability experiment"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "stability", "hybrid"],
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
        help="Number of temporal versions",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate three-way comparison plots",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        help="Path to baseline run summary.json",
    )
    parser.add_argument(
        "--stability-run",
        type=str,
        help="Path to stability run summary.json",
    )
    parser.add_argument(
        "--hybrid-run",
        type=str,
        help="Path to hybrid run summary.json",
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
        if not args.baseline_run or not args.stability_run or not args.hybrid_run:
            print("Error: --compare requires --baseline-run, --stability-run, and --hybrid-run", file=sys.stderr)
            return 1

        with open(args.baseline_run) as f:
            baseline_summary = json.load(f)
        with open(args.stability_run) as f:
            stability_summary = json.load(f)
        with open(args.hybrid_run) as f:
            hybrid_summary = json.load(f)

        output_dir = Path(config.general.output_dir)
        date_suffix = datetime.now().strftime("%Y%m%d")

        generate_three_way_comparison(
            baseline_summary,
            stability_summary,
            hybrid_summary,
            output_dir,
            f"v1.4.0-{date_suffix}",
        )
        return 0

    try:
        summary = run_hybrid_experiment(
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
