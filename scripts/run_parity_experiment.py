#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "matplotlib>=3.8",
# ]
# ///
"""
Parity experiment: Compare offline vs streaming chunking.

Evaluates how closely streaming chunking matches offline chunking
on the same deterministic dataset.

Metrics computed:
- Chunk count parity
- Size distribution parity (Wasserstein-1, KS statistic)
- Boundary placement overlap at various tolerances
- Deviation localization

Usage:
    uv run scripts/run_parity_experiment.py --data-dir eval/data --output-dir eval/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config
from src.chunking.offline import chunk_offline, Chunk
from src.chunking.streaming import StreamingChunker


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
class ChunkBoundaries:
    """Chunk boundary information for a document."""
    doc_id: str
    byte_length: int
    # List of (byte_start, byte_end) for each chunk
    boundaries: list[tuple[int, int]] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    cut_scores: list[float] = field(default_factory=list)

    @property
    def end_offsets(self) -> list[int]:
        """Get end offsets (boundary positions)."""
        return [b[1] for b in self.boundaries]

    @property
    def num_chunks(self) -> int:
        return len(self.boundaries)


@dataclass
class ParityMetrics:
    """Parity metrics between offline and streaming chunking."""
    # Chunk count
    n_offline: int = 0
    n_streaming: int = 0
    delta_pct: float = 0.0

    # Size distribution
    mean_size_offline: float = 0.0
    mean_size_streaming: float = 0.0
    p50_size_offline: float = 0.0
    p50_size_streaming: float = 0.0
    p90_size_offline: float = 0.0
    p90_size_streaming: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0

    # Target alignment
    pct_within_10pct_offline: float = 0.0
    pct_within_10pct_streaming: float = 0.0

    # Boundary overlap at various tolerances
    overlap_tol_0: float = 0.0
    overlap_tol_16: float = 0.0
    overlap_tol_64: float = 0.0
    overlap_tol_256: float = 0.0

    # Precision/Recall/F1
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Deviation localization
    mismatches_near_commit_horizon: float = 0.0
    mismatches_near_domain_boundary: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_count": {
                "n_offline": self.n_offline,
                "n_streaming": self.n_streaming,
                "delta_pct": self.delta_pct,
            },
            "size_distribution": {
                "mean_offline": self.mean_size_offline,
                "mean_streaming": self.mean_size_streaming,
                "p50_offline": self.p50_size_offline,
                "p50_streaming": self.p50_size_streaming,
                "p90_offline": self.p90_size_offline,
                "p90_streaming": self.p90_size_streaming,
                "wasserstein_distance": self.wasserstein_distance,
                "ks_statistic": self.ks_statistic,
                "ks_pvalue": self.ks_pvalue,
            },
            "target_alignment": {
                "pct_within_10pct_offline": self.pct_within_10pct_offline,
                "pct_within_10pct_streaming": self.pct_within_10pct_streaming,
            },
            "boundary_overlap": {
                "tol_0": self.overlap_tol_0,
                "tol_16": self.overlap_tol_16,
                "tol_64": self.overlap_tol_64,
                "tol_256": self.overlap_tol_256,
            },
            "precision_recall": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
            },
            "deviation_localization": {
                "near_commit_horizon": self.mismatches_near_commit_horizon,
                "near_domain_boundary": self.mismatches_near_domain_boundary,
            },
        }


def chunk_document_offline(data: bytes, config: Config) -> ChunkBoundaries:
    """Chunk a document using offline chunking."""
    chunks = chunk_offline(data, config.chunking)
    boundaries = ChunkBoundaries(
        doc_id="",
        byte_length=len(data),
        boundaries=[(c.byte_start, c.byte_end) for c in chunks],
        sizes=[c.byte_end - c.byte_start for c in chunks],
        cut_scores=[c.cut_score for c in chunks],
    )
    return boundaries


def chunk_document_streaming(data: bytes, config: Config, buffer_size: int = 4096) -> ChunkBoundaries:
    """Chunk a document using streaming chunking."""
    chunker = StreamingChunker(config.chunking)

    chunks = []
    offset = 0
    while offset < len(data):
        chunk_data = data[offset:offset + buffer_size]
        for chunk in chunker.feed(chunk_data):
            chunks.append(chunk)
        offset += buffer_size

    # Finalize
    for chunk in chunker.finalize():
        chunks.append(chunk)

    boundaries = ChunkBoundaries(
        doc_id="",
        byte_length=len(data),
        boundaries=[(c.byte_start, c.byte_end) for c in chunks],
        sizes=[c.byte_end - c.byte_start for c in chunks],
        cut_scores=[c.cut_score for c in chunks],
    )
    return boundaries


def compute_boundary_overlap(
    offline_boundaries: list[int],
    streaming_boundaries: list[int],
    tolerance: int,
) -> tuple[int, int, int]:
    """
    Compute boundary overlap at given tolerance.

    Returns:
        (matched_offline, matched_streaming, total_matches)
    """
    matched_offline = 0
    matched_streaming = set()

    for off_b in offline_boundaries:
        for i, str_b in enumerate(streaming_boundaries):
            if abs(off_b - str_b) <= tolerance:
                matched_offline += 1
                matched_streaming.add(i)
                break

    return matched_offline, len(matched_streaming), matched_offline


def compute_parity_metrics(
    offline_results: list[ChunkBoundaries],
    streaming_results: list[ChunkBoundaries],
    config: Config,
    domain_boundaries: list[list[int]] | None = None,
) -> ParityMetrics:
    """Compute parity metrics between offline and streaming chunking."""
    metrics = ParityMetrics()

    # Aggregate all boundaries and sizes
    all_offline_ends = []
    all_streaming_ends = []
    all_offline_sizes = []
    all_streaming_sizes = []

    for off, strm in zip(offline_results, streaming_results):
        all_offline_ends.extend(off.end_offsets)
        all_streaming_ends.extend(strm.end_offsets)
        all_offline_sizes.extend(off.sizes)
        all_streaming_sizes.extend(strm.sizes)

    # Chunk count
    metrics.n_offline = len(all_offline_sizes)
    metrics.n_streaming = len(all_streaming_sizes)
    if metrics.n_offline > 0:
        metrics.delta_pct = 100.0 * (metrics.n_streaming - metrics.n_offline) / metrics.n_offline

    if not all_offline_sizes or not all_streaming_sizes:
        return metrics

    # Size distribution
    off_sizes = np.array(all_offline_sizes)
    strm_sizes = np.array(all_streaming_sizes)

    metrics.mean_size_offline = float(np.mean(off_sizes))
    metrics.mean_size_streaming = float(np.mean(strm_sizes))
    metrics.p50_size_offline = float(np.percentile(off_sizes, 50))
    metrics.p50_size_streaming = float(np.percentile(strm_sizes, 50))
    metrics.p90_size_offline = float(np.percentile(off_sizes, 90))
    metrics.p90_size_streaming = float(np.percentile(strm_sizes, 90))

    # Wasserstein-1 distance
    metrics.wasserstein_distance = float(stats.wasserstein_distance(off_sizes, strm_sizes))

    # KS statistic
    ks_stat, ks_pval = stats.ks_2samp(off_sizes, strm_sizes)
    metrics.ks_statistic = float(ks_stat)
    metrics.ks_pvalue = float(ks_pval)

    # Target alignment
    L_target = config.chunking.L_target_bytes
    pct_threshold = 0.10

    within_offline = np.sum(np.abs(off_sizes - L_target) / L_target <= pct_threshold)
    within_streaming = np.sum(np.abs(strm_sizes - L_target) / L_target <= pct_threshold)
    metrics.pct_within_10pct_offline = 100.0 * within_offline / len(off_sizes)
    metrics.pct_within_10pct_streaming = 100.0 * within_streaming / len(strm_sizes)

    # Boundary overlap at various tolerances
    for tol in [0, 16, 64, 256]:
        matched_off, matched_strm, _ = compute_boundary_overlap(
            all_offline_ends, all_streaming_ends, tol
        )
        overlap = matched_off / len(all_offline_ends) if all_offline_ends else 0.0
        if tol == 0:
            metrics.overlap_tol_0 = overlap
        elif tol == 16:
            metrics.overlap_tol_16 = overlap
        elif tol == 64:
            metrics.overlap_tol_64 = overlap
        elif tol == 256:
            metrics.overlap_tol_256 = overlap

    # Precision/Recall/F1 at tol=64
    matched_off, matched_strm, _ = compute_boundary_overlap(
        all_offline_ends, all_streaming_ends, 64
    )
    metrics.precision = matched_strm / len(all_streaming_ends) if all_streaming_ends else 0.0
    metrics.recall = matched_off / len(all_offline_ends) if all_offline_ends else 0.0
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    # Deviation localization
    commit_horizon = config.chunking.commit_horizon_bytes
    unmatched_offline = set(all_offline_ends)
    unmatched_streaming = set(all_streaming_ends)

    # Find unmatched boundaries
    for off_b in all_offline_ends:
        for str_b in all_streaming_ends:
            if abs(off_b - str_b) <= 64:
                unmatched_offline.discard(off_b)
                unmatched_streaming.discard(str_b)
                break

    # Check if mismatches are near commit horizon boundaries
    near_horizon = 0
    for mis in unmatched_offline | unmatched_streaming:
        # Check if mismatch is near a streaming boundary (within commit_horizon)
        for str_b in all_streaming_ends:
            if abs(mis - str_b) <= commit_horizon and abs(mis - str_b) > 64:
                near_horizon += 1
                break

    total_mismatches = len(unmatched_offline) + len(unmatched_streaming)
    if total_mismatches > 0:
        metrics.mismatches_near_commit_horizon = near_horizon / total_mismatches

    # Domain boundary localization (if available)
    if domain_boundaries:
        all_domain_bounds = []
        for db in domain_boundaries:
            all_domain_bounds.extend(db)

        near_domain = 0
        for mis in unmatched_offline | unmatched_streaming:
            for db in all_domain_bounds:
                if abs(mis - db) <= 256:
                    near_domain += 1
                    break

        if total_mismatches > 0:
            metrics.mismatches_near_domain_boundary = near_domain / total_mismatches

    return metrics


def generate_parity_plots(
    offline_results: list[ChunkBoundaries],
    streaming_results: list[ChunkBoundaries],
    metrics: ParityMetrics,
    output_dir: Path,
) -> None:
    """Generate parity comparison plots."""
    # Set paper style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 1. Size distribution comparison
    fig, ax = plt.subplots()
    all_offline_sizes = []
    all_streaming_sizes = []
    for off, strm in zip(offline_results, streaming_results):
        all_offline_sizes.extend(off.sizes)
        all_streaming_sizes.extend(strm.sizes)

    bins = np.linspace(0, max(max(all_offline_sizes), max(all_streaming_sizes)), 50)
    ax.hist(all_offline_sizes, bins=bins, alpha=0.6, label=f"Offline (n={len(all_offline_sizes)})", color="steelblue")
    ax.hist(all_streaming_sizes, bins=bins, alpha=0.6, label=f"Streaming (n={len(all_streaming_sizes)})", color="coral")
    ax.axvline(2048, color="gray", linestyle="--", linewidth=1, label="L_target")
    ax.set_xlabel("Chunk Size (bytes)")
    ax.set_ylabel("Count")
    ax.set_title("Chunk Size Distribution: Offline vs Streaming")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "chunk_parity_sizes.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Boundary overlap by tolerance
    fig, ax = plt.subplots()
    tolerances = [0, 16, 64, 256]
    overlaps = [
        metrics.overlap_tol_0,
        metrics.overlap_tol_16,
        metrics.overlap_tol_64,
        metrics.overlap_tol_256,
    ]
    bars = ax.bar(range(len(tolerances)), [o * 100 for o in overlaps], color="steelblue")
    ax.set_xticks(range(len(tolerances)))
    ax.set_xticklabels([f"±{t}" for t in tolerances])
    ax.set_xlabel("Tolerance (bytes)")
    ax.set_ylabel("Boundary Overlap (%)")
    ax.set_title("Boundary Placement Overlap at Various Tolerances")
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, val in zip(bars, overlaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "chunk_parity_overlap.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_parity_table(metrics: ParityMetrics, output_path: Path) -> None:
    """Generate LaTeX table for parity metrics."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Streaming Chunking Parity with Offline Chunking}
\label{tab:streaming-parity}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Offline} & \textbf{Streaming} \\
\midrule
\multicolumn{3}{l}{\textit{Chunk Count}} \\
Total Chunks & """ + f"{metrics.n_offline}" + r""" & """ + f"{metrics.n_streaming}" + r""" \\
Delta & \multicolumn{2}{c}{""" + f"{metrics.delta_pct:+.1f}\\%" + r"""} \\
\midrule
\multicolumn{3}{l}{\textit{Size Distribution (bytes)}} \\
Mean & """ + f"{metrics.mean_size_offline:.0f}" + r""" & """ + f"{metrics.mean_size_streaming:.0f}" + r""" \\
P50 & """ + f"{metrics.p50_size_offline:.0f}" + r""" & """ + f"{metrics.p50_size_streaming:.0f}" + r""" \\
P90 & """ + f"{metrics.p90_size_offline:.0f}" + r""" & """ + f"{metrics.p90_size_streaming:.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Target Alignment ($\pm$10\% of L\_target)}} \\
\% Within Target & """ + f"{metrics.pct_within_10pct_offline:.1f}\\%" + r""" & """ + f"{metrics.pct_within_10pct_streaming:.1f}\\%" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Boundary Overlap (recall)}} \\
Exact ($\pm$0 bytes) & \multicolumn{2}{c}{""" + f"{metrics.overlap_tol_0*100:.1f}\\%" + r"""} \\
$\pm$16 bytes & \multicolumn{2}{c}{""" + f"{metrics.overlap_tol_16*100:.1f}\\%" + r"""} \\
$\pm$64 bytes & \multicolumn{2}{c}{""" + f"{metrics.overlap_tol_64*100:.1f}\\%" + r"""} \\
$\pm$256 bytes & \multicolumn{2}{c}{""" + f"{metrics.overlap_tol_256*100:.1f}\\%" + r"""} \\
\midrule
\multicolumn{3}{l}{\textit{Distribution Distance}} \\
Wasserstein-1 & \multicolumn{2}{c}{""" + f"{metrics.wasserstein_distance:.1f}" + r""" bytes} \\
KS Statistic & \multicolumn{2}{c}{""" + f"{metrics.ks_statistic:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    output_path.write_text(latex)


def run_parity_experiment(
    data_dir: Path,
    output_dir: Path,
    config: Config,
    run_name_base: str,
) -> dict:
    """Run the full parity experiment."""
    logger = logging.getLogger(__name__)

    git_version = get_git_version()
    timestamp = datetime.now(timezone.utc)

    # Create output directories
    offline_dir = output_dir / f"{run_name_base}-offline"
    streaming_dir = output_dir / f"{run_name_base}-streaming"
    comparison_dir = output_dir / "comparisons" / run_name_base

    offline_dir.mkdir(parents=True, exist_ok=True)
    streaming_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load data manifest
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        data_manifest = json.load(f)

    doc_ids = data_manifest.get("doc_ids", data_manifest.get("document_ids", []))
    corpus_dir = data_dir / "corpus"

    logger.info(f"Processing {len(doc_ids)} documents")

    offline_results = []
    streaming_results = []
    domain_boundaries = []

    for doc_id in doc_ids:
        # Find the file
        content = None
        for ext in [".txt", ".py", ".json", ".log"]:
            fpath = corpus_dir / f"{doc_id}{ext}"
            if fpath.exists():
                content = fpath.read_bytes()
                break

        if content is None:
            logger.warning(f"Document not found: {doc_id}")
            continue

        # Run offline chunking
        off_result = chunk_document_offline(content, config)
        off_result.doc_id = doc_id
        offline_results.append(off_result)

        # Run streaming chunking
        strm_result = chunk_document_streaming(content, config, buffer_size=4096)
        strm_result.doc_id = doc_id
        streaming_results.append(strm_result)

        # Get domain boundaries if available
        # (would need to load from anchors.json or similar)
        domain_boundaries.append([])  # Placeholder

    # Compute parity metrics
    logger.info("Computing parity metrics")
    metrics = compute_parity_metrics(
        offline_results, streaming_results, config, domain_boundaries
    )

    # Save offline results
    offline_summary = {
        "run_name": f"{run_name_base}-offline",
        "mode": "offline",
        "git_version": git_version,
        "timestamp": timestamp.isoformat(),
        "config": config.to_dict(),
        "n_documents": len(offline_results),
        "n_chunks": metrics.n_offline,
        "mean_chunk_size": metrics.mean_size_offline,
        "p50_chunk_size": metrics.p50_size_offline,
        "p90_chunk_size": metrics.p90_size_offline,
        "pct_within_10pct_target": metrics.pct_within_10pct_offline,
    }
    with open(offline_dir / "summary.json", "w") as f:
        json.dump(offline_summary, f, indent=2)

    # Save per-doc boundaries for offline
    offline_boundaries_data = [
        {"doc_id": r.doc_id, "boundaries": r.boundaries, "sizes": r.sizes}
        for r in offline_results
    ]
    with open(offline_dir / "boundaries.json", "w") as f:
        json.dump(offline_boundaries_data, f, indent=2)

    # Save streaming results
    streaming_summary = {
        "run_name": f"{run_name_base}-streaming",
        "mode": "streaming",
        "git_version": git_version,
        "timestamp": timestamp.isoformat(),
        "config": config.to_dict(),
        "n_documents": len(streaming_results),
        "n_chunks": metrics.n_streaming,
        "mean_chunk_size": metrics.mean_size_streaming,
        "p50_chunk_size": metrics.p50_size_streaming,
        "p90_chunk_size": metrics.p90_size_streaming,
        "pct_within_10pct_target": metrics.pct_within_10pct_streaming,
    }
    with open(streaming_dir / "summary.json", "w") as f:
        json.dump(streaming_summary, f, indent=2)

    # Save per-doc boundaries for streaming
    streaming_boundaries_data = [
        {"doc_id": r.doc_id, "boundaries": r.boundaries, "sizes": r.sizes}
        for r in streaming_results
    ]
    with open(streaming_dir / "boundaries.json", "w") as f:
        json.dump(streaming_boundaries_data, f, indent=2)

    # Save comparison summary
    comparison_summary = {
        "run_name": run_name_base,
        "git_version": git_version,
        "timestamp": timestamp.isoformat(),
        "config": config.to_dict(),
        "metrics": metrics.to_dict(),
    }
    with open(comparison_dir / "summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)

    # Generate plots
    logger.info("Generating plots")
    generate_parity_plots(offline_results, streaming_results, metrics, comparison_dir)

    # Generate LaTeX table
    logger.info("Generating LaTeX table")
    generate_parity_table(metrics, comparison_dir / "parity_table.tex")

    # Copy to paper directories
    paper_tables = Path("paper/tables")
    paper_figures = Path("paper/figures")
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(comparison_dir / "parity_table.tex", paper_tables / f"chunk_parity_{run_name_base}.tex")
    shutil.copy(comparison_dir / "chunk_parity_sizes.pdf", paper_figures / f"chunk_parity_sizes_{run_name_base}.pdf")
    shutil.copy(comparison_dir / "chunk_parity_overlap.pdf", paper_figures / f"chunk_parity_overlap_{run_name_base}.pdf")

    logger.info(f"Parity experiment complete. Results in {comparison_dir}")

    return {
        "run_name": run_name_base,
        "metrics": metrics.to_dict(),
        "offline_dir": str(offline_dir),
        "streaming_dir": str(streaming_dir),
        "comparison_dir": str(comparison_dir),
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run offline vs streaming chunking parity experiment"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("eval/data"),
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Base name for this run (auto-generated if not provided)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_config(args.config) if args.config.exists() else Config()

    # Generate run name if not provided
    if args.run_name is None:
        date_str = datetime.now().strftime("%Y%m%d")
        args.run_name = f"v1.2.0-parity-{date_str}"

    logger.info(f"Starting parity experiment: {args.run_name}")
    logger.info(f"Diagnostics mode: {config.diagnostics.mode}")

    try:
        result = run_parity_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            run_name_base=args.run_name,
        )

        # Print summary
        print("\n" + "="*60)
        print("PARITY EXPERIMENT RESULTS")
        print("="*60)
        m = result["metrics"]
        print(f"\nChunk Count:")
        print(f"  Offline:   {m['chunk_count']['n_offline']}")
        print(f"  Streaming: {m['chunk_count']['n_streaming']}")
        print(f"  Delta:     {m['chunk_count']['delta_pct']:+.1f}%")

        print(f"\nSize Distribution:")
        print(f"  Mean:  {m['size_distribution']['mean_offline']:.0f} (off) vs {m['size_distribution']['mean_streaming']:.0f} (str)")
        print(f"  P50:   {m['size_distribution']['p50_offline']:.0f} (off) vs {m['size_distribution']['p50_streaming']:.0f} (str)")
        print(f"  P90:   {m['size_distribution']['p90_offline']:.0f} (off) vs {m['size_distribution']['p90_streaming']:.0f} (str)")
        print(f"  Wasserstein: {m['size_distribution']['wasserstein_distance']:.1f} bytes")

        print(f"\nTarget Alignment (±10% of L_target):")
        print(f"  Offline:   {m['target_alignment']['pct_within_10pct_offline']:.1f}%")
        print(f"  Streaming: {m['target_alignment']['pct_within_10pct_streaming']:.1f}%")

        print(f"\nBoundary Overlap:")
        print(f"  ±0 bytes:   {m['boundary_overlap']['tol_0']*100:.1f}%")
        print(f"  ±16 bytes:  {m['boundary_overlap']['tol_16']*100:.1f}%")
        print(f"  ±64 bytes:  {m['boundary_overlap']['tol_64']*100:.1f}%")
        print(f"  ±256 bytes: {m['boundary_overlap']['tol_256']*100:.1f}%")

        print(f"\nPrecision/Recall/F1 (tol=64):")
        print(f"  Precision: {m['precision_recall']['precision']*100:.1f}%")
        print(f"  Recall:    {m['precision_recall']['recall']*100:.1f}%")
        print(f"  F1:        {m['precision_recall']['f1']*100:.1f}%")

        print("\n" + "="*60)

        return 0

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
