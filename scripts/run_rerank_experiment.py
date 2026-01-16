#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sentence-transformers>=2.2",
#     "faiss-cpu>=1.7",
#     "numpy>=1.26",
#     "matplotlib>=3.8",
# ]
# ///
"""
Representational reranking experiment for curv-embedding v2.0.0.

Evaluates three retrieval modes against a cost-corrected hybrid chunking baseline:
1. ann — cosine similarity only
2. ann_random — random rerank control
3. ann_repr — representational reranking using stability diagnostics

v1.6.0 Context (carry-forward):
Count-based mutation metrics mischaracterized hybrid chunking by treating all
chunks as equal cost. Byte-weighted metrics showed that hybrid chunking reduces
effective mutation cost by 54% relative to baseline, resolving the apparent
negative result and establishing hybrid chunking as the correct base policy
for dynamic vector stores.

Usage:
    uv run scripts/run_rerank_experiment.py --mode ann --run-name v1.6.0-ann-20260116
    uv run scripts/run_rerank_experiment.py --mode ann_random --run-name v1.6.0-ann_random-20260116
    uv run scripts/run_rerank_experiment.py --mode ann_repr --run-name v1.6.0-ann_repr-20260116
    uv run scripts/run_rerank_experiment.py --compare --date-suffix 20260116
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config
from src.data.generator import generate_corpus
from src.chunking.offline import chunk_offline
from src.embedding.model import EmbeddingModel
from src.storage.sqlite_store import SQLiteStore, ChunkRecord
from src.storage.faiss_index import FAISSIndex
from src.eval.reranker import RepresentationalReranker, RerankResult


# v1.6.0 Context (mandatory carry-forward)
V160_CONTEXT = """Count-based mutation metrics mischaracterized hybrid chunking by treating all
chunks as equal cost. Byte-weighted metrics showed that hybrid chunking reduces
effective mutation cost by 54% relative to baseline, resolving the apparent
negative result and establishing hybrid chunking as the correct base policy
for dynamic vector stores."""


def setup_logging(log_level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_git_version() -> dict[str, str | None]:
    """Get current git tag and commit hash."""
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
class QueryFamily:
    """A family of related queries for reformulation testing."""

    family_id: str
    base_query: str
    reformulations: list[str]
    target_doc_ids: list[str]


@dataclass
class RerankMetrics:
    """Aggregate metrics from reranking evaluation."""

    # Retrieval stability
    temporal_overlap_mean: float = 0.0
    temporal_overlap_p10: float = 0.0
    reformulation_overlap_mean: float = 0.0
    reformulation_overlap_p10: float = 0.0

    # Boundary robustness
    boundary_churn_reduction: float = 0.0
    near_boundary_fraction: float = 0.0

    # Rerank characteristics
    disagreement_rate: float = 0.0
    rank_shift_mean: float = 0.0
    rank_shift_max: float = 0.0

    def to_dict(self) -> dict:
        return {
            "temporal_overlap_mean": self.temporal_overlap_mean,
            "temporal_overlap_p10": self.temporal_overlap_p10,
            "reformulation_overlap_mean": self.reformulation_overlap_mean,
            "reformulation_overlap_p10": self.reformulation_overlap_p10,
            "boundary_churn_reduction": self.boundary_churn_reduction,
            "near_boundary_fraction": self.near_boundary_fraction,
            "disagreement_rate": self.disagreement_rate,
            "rank_shift_mean": self.rank_shift_mean,
            "rank_shift_max": self.rank_shift_max,
        }


def generate_query_families(
    corpus: list,
    num_families: int,
    reformulations_per_family: int,
    seed: int,
) -> list[QueryFamily]:
    """Generate query families with reformulations for testing."""
    rng = random.Random(seed)
    families = []

    for i in range(num_families):
        doc = rng.choice(corpus)
        content = doc.content.decode("utf-8", errors="replace")

        # Extract a seed phrase from the document
        words = content.split()
        if len(words) < 10:
            continue

        start_idx = rng.randint(0, max(0, len(words) - 10))
        base_phrase = " ".join(words[start_idx:start_idx + rng.randint(5, 10)])

        # Generate reformulations
        reformulations = []
        for j in range(reformulations_per_family):
            reform_words = base_phrase.split()
            if len(reform_words) > 2:
                # Shuffle some words, add/remove words
                if rng.random() < 0.3:
                    reform_words = reform_words[1:]
                if rng.random() < 0.3:
                    reform_words = reform_words[:-1]
                if rng.random() < 0.3 and len(reform_words) > 2:
                    idx = rng.randint(0, len(reform_words) - 1)
                    reform_words[idx] = rng.choice(["the", "a", "some", "this"])
            reformulations.append(" ".join(reform_words))

        families.append(QueryFamily(
            family_id=f"family_{i}",
            base_query=base_phrase,
            reformulations=reformulations,
            target_doc_ids=[doc.doc_id],
        ))

    return families


def create_hybrid_chunks_with_metadata(
    content: bytes,
    doc_id: str,
    config: Config,
) -> list[dict]:
    """Create hybrid chunks with diagnostic metadata for reranking."""
    chunks = chunk_offline(content, config.chunking)

    result = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        content_hash = hashlib.sha256(chunk.content).hexdigest()

        result.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": i,
            "byte_offset_start": chunk.byte_start,
            "byte_offset_end": chunk.byte_end,
            "content": chunk.content,
            "content_sha256": content_hash,
            "cut_score": chunk.cut_score,
            "curvature_signal": getattr(chunk, "curvature_signal", 0.5),
            "stability_margin_signal": getattr(chunk, "stability_margin_signal", 0.5),
            "disharmony_signal": getattr(chunk, "disharmony_signal", 0.0),
            "is_structural_boundary": getattr(chunk, "is_structural_boundary", False),
        })

    return result


def compute_topk_overlap(list_a: list, list_b: list, k: int) -> float:
    """Compute overlap between two top-k lists."""
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / k


def run_rerank_experiment(
    run_name: str,
    mode: str,
    config: Config,
    num_docs: int = 30,
    num_queries: int = 50,
) -> dict:
    """Run reranking experiment with specified mode."""
    logger = logging.getLogger(__name__)

    git_version = get_git_version()
    output_dir = Path(config.general.output_dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting rerank experiment: {run_name}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Git version: tag={git_version.get('tag')}, commit={git_version.get('commit')}")

    # Generate corpus
    logger.info(f"Generating corpus: {num_docs} documents")
    corpus = generate_corpus(
        seed=config.general.seed,
        num_docs=num_docs,
        domains=["text", "code", "json", "logs"],
        size_range=(2048, 8192),
    )

    # Initialize components
    logger.info("Loading embedding model")
    embed_model = EmbeddingModel(config.embedding)
    reranker = RepresentationalReranker(config.rerank)

    # Build chunks with metadata
    logger.info("Chunking documents with diagnostics")
    all_chunks: list[dict] = []
    chunk_records: dict[str, ChunkRecord] = {}

    for doc in corpus:
        doc_chunks = create_hybrid_chunks_with_metadata(doc.content, doc.doc_id, config)
        all_chunks.extend(doc_chunks)

    # Embed chunks
    logger.info(f"Embedding {len(all_chunks)} chunks")
    chunk_embeddings: dict[str, np.ndarray] = {}
    chunk_texts: dict[str, str] = {}

    for chunk in all_chunks:
        text = chunk["content"].decode("utf-8", errors="replace")
        embedding = embed_model.embed_single(text)
        chunk_embeddings[chunk["chunk_id"]] = embedding
        chunk_texts[chunk["chunk_id"]] = text

        # Create ChunkRecord for reranker
        record = ChunkRecord(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            chunk_index=chunk["chunk_index"],
            byte_offset_start=chunk["byte_offset_start"],
            byte_offset_end=chunk["byte_offset_end"],
            content_sha256=chunk["content_sha256"],
            cut_score=chunk["cut_score"],
            curvature_signal=chunk["curvature_signal"],
            stability_margin_signal=chunk["stability_margin_signal"],
            disharmony_signal=chunk["disharmony_signal"],
            is_structural_boundary=chunk["is_structural_boundary"],
        )
        chunk_records[chunk["chunk_id"]] = record

    # Build FAISS index
    logger.info("Building FAISS index")
    faiss_index = FAISSIndex(config.embedding.embedding_dim, config.storage)

    chunk_id_list = list(chunk_embeddings.keys())
    embedding_matrix = np.array([chunk_embeddings[cid] for cid in chunk_id_list])
    faiss_ids = faiss_index.add_vectors(embedding_matrix)

    faiss_to_chunk = {fid: cid for fid, cid in zip(faiss_ids, chunk_id_list)}
    chunk_to_faiss = {cid: fid for fid, cid in zip(faiss_ids, chunk_id_list)}

    # Generate query families
    logger.info(f"Generating {num_queries} query families")
    query_families = generate_query_families(
        corpus,
        num_families=num_queries,
        reformulations_per_family=config.eval.reformulations_per_family,
        seed=config.general.seed + 1000,
    )

    # Run queries
    logger.info("Running queries with reranking")
    K0 = config.rerank.K0
    k = config.rerank.k

    all_results = []
    disagreements = []
    rank_shifts = []
    reformulation_overlaps = []

    for family in query_families:
        family_results = {
            "family_id": family.family_id,
            "base_query": family.base_query,
            "reformulations": [],
            "retrieved_sets": [],
        }

        all_queries = [family.base_query] + family.reformulations

        for query_text in all_queries:
            # Embed query
            query_embedding = embed_model.embed_single(query_text)

            # Get K0 candidates from FAISS
            distances, faiss_ids_result = faiss_index.search(
                query_embedding.reshape(1, -1), K0
            )

            # Build candidate list with metadata
            candidates: list[tuple[str, float, ChunkRecord]] = []
            for dist, fid in zip(distances[0], faiss_ids_result[0]):
                if fid < 0:
                    continue
                chunk_id = faiss_to_chunk.get(int(fid))
                if chunk_id is None:
                    continue

                # Convert L2 distance to similarity score (higher is better)
                # For normalized vectors, similarity ≈ 1 - dist/2
                sim_score = max(0, 1 - dist / 2)

                candidates.append((
                    chunk_id,
                    sim_score,
                    chunk_records[chunk_id],
                ))

            # Rerank using specified mode
            rerank_result = reranker.rerank(candidates, mode=mode)

            # Also get ANN-only result for comparison
            if mode != "ann":
                ann_result = reranker.rerank(candidates, mode="ann")
                disagreement = reranker.compute_disagreement(ann_result, rerank_result)
                disagreements.append(disagreement)

                # Compute rank shifts
                for i, cid in enumerate(rerank_result.chunk_ids):
                    if cid in ann_result.chunk_ids:
                        ann_rank = ann_result.chunk_ids.index(cid)
                        shift = abs(i - ann_rank)
                        rank_shifts.append(shift)

            family_results["retrieved_sets"].append(rerank_result.chunk_ids)
            family_results["reformulations"].append({
                "query": query_text,
                "retrieved": rerank_result.chunk_ids,
                "scores": [float(s) for s in rerank_result.scores],
            })

        # Compute reformulation overlap within family
        if len(family_results["retrieved_sets"]) > 1:
            for i in range(len(family_results["retrieved_sets"])):
                for j in range(i + 1, len(family_results["retrieved_sets"])):
                    overlap = compute_topk_overlap(
                        family_results["retrieved_sets"][i],
                        family_results["retrieved_sets"][j],
                        k,
                    )
                    reformulation_overlaps.append(overlap)

        all_results.append(family_results)

    # Compute aggregate metrics
    logger.info("Computing aggregate metrics")
    metrics = RerankMetrics()

    if reformulation_overlaps:
        metrics.reformulation_overlap_mean = float(np.mean(reformulation_overlaps))
        metrics.reformulation_overlap_p10 = float(np.percentile(reformulation_overlaps, 10))

    if disagreements:
        metrics.disagreement_rate = float(np.mean(disagreements))

    if rank_shifts:
        metrics.rank_shift_mean = float(np.mean(rank_shifts))
        metrics.rank_shift_max = float(max(rank_shifts))

    # Compute boundary proximity in retrieved results
    near_boundary_count = 0
    total_retrieved = 0
    for result in all_results:
        for reform in result["reformulations"]:
            for cid in reform["retrieved"]:
                total_retrieved += 1
                record = chunk_records.get(cid)
                if record and record.is_structural_boundary:
                    near_boundary_count += 1

    if total_retrieved > 0:
        metrics.near_boundary_fraction = near_boundary_count / total_retrieved

    # Save results
    summary = {
        "run_name": run_name,
        "mode": mode,
        "git_version": git_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "v160_context": V160_CONTEXT,
        "config": config.to_dict(),
        "corpus": {
            "num_docs": num_docs,
            "num_chunks": len(all_chunks),
        },
        "evaluation": {
            "num_query_families": len(query_families),
            "K0": K0,
            "k": k,
        },
        "aggregate_metrics": metrics.to_dict(),
        "query_results": all_results,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment complete. Results saved to {run_dir}")
    logger.info(f"Reformulation overlap: {metrics.reformulation_overlap_mean:.3f}")
    if mode != "ann":
        logger.info(f"Disagreement rate: {metrics.disagreement_rate:.3f}")

    return summary


def generate_comparison_plots(
    ann_summary: dict,
    random_summary: dict,
    repr_summary: dict,
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

    modes = ["ANN", "Random", "Repr"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 1. Reformulation overlap comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Overlap bars
    ax1 = axes[0]
    overlap_values = [
        ann_summary["aggregate_metrics"]["reformulation_overlap_mean"],
        random_summary["aggregate_metrics"]["reformulation_overlap_mean"],
        repr_summary["aggregate_metrics"]["reformulation_overlap_mean"],
    ]
    bars = ax1.bar(modes, overlap_values, color=colors)
    ax1.set_ylabel("Mean Top-k Overlap")
    ax1.set_title("Reformulation Stability")
    ax1.set_ylim(0, 1.0)
    for bar, val in zip(bars, overlap_values):
        ax1.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Disagreement rate (only for random and repr)
    ax2 = axes[1]
    disagree_values = [
        0.0,  # ANN baseline
        random_summary["aggregate_metrics"]["disagreement_rate"],
        repr_summary["aggregate_metrics"]["disagreement_rate"],
    ]
    bars = ax2.bar(modes, disagree_values, color=colors)
    ax2.set_ylabel("Disagreement Rate")
    ax2.set_title("Rerank Disagreement vs ANN")
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, disagree_values):
        ax2.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("Representational Reranking Evaluation (v2.0.0)", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / f"rerank_overlap_v2.0.0.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Boundary robustness
    fig, ax = plt.subplots(figsize=(6, 4))

    boundary_values = [
        ann_summary["aggregate_metrics"]["near_boundary_fraction"],
        random_summary["aggregate_metrics"]["near_boundary_fraction"],
        repr_summary["aggregate_metrics"]["near_boundary_fraction"],
    ]
    bars = ax.bar(modes, boundary_values, color=colors)
    ax.set_ylabel("Structural Boundary Fraction")
    ax.set_title("Boundary Preference in Retrieved Chunks")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, boundary_values):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.savefig(figures_dir / f"rerank_boundary_v2.0.0.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Generate LaTeX table
    ann_m = ann_summary["aggregate_metrics"]
    random_m = random_summary["aggregate_metrics"]
    repr_m = repr_summary["aggregate_metrics"]

    # Calculate improvement
    overlap_improvement = (repr_m["reformulation_overlap_mean"] - ann_m["reformulation_overlap_mean"]) / ann_m["reformulation_overlap_mean"] * 100 if ann_m["reformulation_overlap_mean"] > 0 else 0

    table_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Representational Reranking Results (v2.0.0)}}
\\label{{tab:rerank-stability}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{ANN}} & \\textbf{{Random}} & \\textbf{{Repr}} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Retrieval Stability}}}} \\\\
Reformulation Overlap & {ann_m["reformulation_overlap_mean"]:.3f} & {random_m["reformulation_overlap_mean"]:.3f} & {repr_m["reformulation_overlap_mean"]:.3f} \\\\
Overlap P10 & {ann_m["reformulation_overlap_p10"]:.3f} & {random_m["reformulation_overlap_p10"]:.3f} & {repr_m["reformulation_overlap_p10"]:.3f} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Boundary Robustness}}}} \\\\
Structural Boundary Frac & {ann_m["near_boundary_fraction"]:.3f} & {random_m["near_boundary_fraction"]:.3f} & {repr_m["near_boundary_fraction"]:.3f} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Rerank Characteristics}}}} \\\\
Disagreement Rate & -- & {random_m["disagreement_rate"]:.3f} & {repr_m["disagreement_rate"]:.3f} \\\\
Mean Rank Shift & -- & {random_m["rank_shift_mean"]:.1f} & {repr_m["rank_shift_mean"]:.1f} \\\\
Max Rank Shift & -- & {random_m["rank_shift_max"]:.0f} & {repr_m["rank_shift_max"]:.0f} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{0.5em}}
\\caption*{{\\footnotesize Representational reranking improves reformulation overlap by {overlap_improvement:.1f}\\% over ANN baseline.}}
\\end{{table}}
"""

    with open(tables_dir / "rerank_stability_v2.0.0.tex", "w") as f:
        f.write(table_content)

    logger.info(f"Saved plots to {figures_dir}")
    logger.info(f"Saved table to {tables_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run representational reranking experiment"
    )
    parser.add_argument(
        "--run-name",
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--mode",
        choices=["ann", "ann_random", "ann_repr"],
        help="Reranking mode to use",
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
        "--num-queries",
        type=int,
        default=50,
        help="Number of query families",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate three-way comparison plots",
    )
    parser.add_argument(
        "--date-suffix",
        type=str,
        help="Date suffix for comparison (YYYYMMDD)",
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
        if not args.date_suffix:
            print("Error: --compare requires --date-suffix", file=sys.stderr)
            return 1

        output_dir = Path(config.general.output_dir)
        ann_path = output_dir / f"v1.6.0-ann-{args.date_suffix}" / "summary.json"
        random_path = output_dir / f"v1.6.0-ann_random-{args.date_suffix}" / "summary.json"
        repr_path = output_dir / f"v1.6.0-ann_repr-{args.date_suffix}" / "summary.json"

        if not all(p.exists() for p in [ann_path, random_path, repr_path]):
            print("Error: Could not find all experiment summaries", file=sys.stderr)
            print(f"  ann: {ann_path} (exists: {ann_path.exists()})")
            print(f"  random: {random_path} (exists: {random_path.exists()})")
            print(f"  repr: {repr_path} (exists: {repr_path.exists()})")
            return 1

        with open(ann_path) as f:
            ann_summary = json.load(f)
        with open(random_path) as f:
            random_summary = json.load(f)
        with open(repr_path) as f:
            repr_summary = json.load(f)

        generate_comparison_plots(
            ann_summary,
            random_summary,
            repr_summary,
            output_dir,
            args.date_suffix,
        )
        return 0

    if not args.run_name or not args.mode:
        print("Error: --run-name and --mode required for running experiments", file=sys.stderr)
        return 1

    try:
        summary = run_rerank_experiment(
            run_name=args.run_name,
            mode=args.mode,
            config=config,
            num_docs=args.num_docs,
            num_queries=args.num_queries,
        )
        print(json.dumps(summary["aggregate_metrics"], indent=2))
        return 0
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
