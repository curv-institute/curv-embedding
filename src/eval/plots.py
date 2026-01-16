"""
Plotting utilities for curv-embedding evaluation results.

Generates paper-quality matplotlib figures for drift, churn, overlap,
chunk size distribution, and boundary sensitivity analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def set_paper_style() -> None:
    """Configure matplotlib for paper-quality figures.

    Sets font sizes, figure dimensions, and styling appropriate for
    academic publications. Removes excessive decoration.
    """
    plt.rcParams.update({
        # Font settings
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Figure settings
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # Axes settings
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Legend settings
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",

        # Remove top and right spines
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Use LaTeX-like rendering
        "text.usetex": False,  # Set True if LaTeX is available
        "mathtext.fontset": "stix",
    })


def plot_drift_distribution(
    drift_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot histogram of drift values comparing baseline vs stability-driven.

    Args:
        drift_results: Dictionary containing drift measurements with keys:
            - 'baseline': list of drift values for baseline chunking
            - 'stability': list of drift values for stability-driven chunking
            Optionally includes 'baseline_mean', 'stability_mean', etc.
        output_path: Path to save the PDF figure.
    """
    set_paper_style()

    baseline_drifts = np.array(drift_results.get("baseline", []))
    stability_drifts = np.array(drift_results.get("stability", []))

    fig, ax = plt.subplots(figsize=(6, 4))

    # Determine common bins for both histograms
    all_drifts = np.concatenate([baseline_drifts, stability_drifts]) if len(baseline_drifts) > 0 and len(stability_drifts) > 0 else np.array([0, 1])
    bins = np.linspace(0, max(all_drifts.max(), 1.0), 31)

    # Plot histograms
    if len(baseline_drifts) > 0:
        ax.hist(
            baseline_drifts,
            bins=bins,
            alpha=0.7,
            label=f"Baseline (mean={baseline_drifts.mean():.3f})",
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
        )

    if len(stability_drifts) > 0:
        ax.hist(
            stability_drifts,
            bins=bins,
            alpha=0.7,
            label=f"Stability-driven (mean={stability_drifts.mean():.3f})",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Embedding Drift (L2 distance)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Embedding Drift After Updates")
    ax.legend(loc="upper right")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_churn_over_updates(
    churn_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot line chart of neighbor churn over sequential updates.

    Args:
        churn_results: List of dictionaries, one per update, each containing:
            - 'update_idx': int, the update sequence number
            - 'baseline_churn': float, churn rate for baseline
            - 'stability_churn': float, churn rate for stability-driven
            Optionally includes 'baseline_churn_std', 'stability_churn_std'.
        output_path: Path to save the PDF figure.
    """
    set_paper_style()

    if not churn_results:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No churn data available", ha="center", va="center", transform=ax.transAxes)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        return

    updates = [r.get("update_idx", i) for i, r in enumerate(churn_results)]
    baseline_churn = [r.get("baseline_churn", 0) for r in churn_results]
    stability_churn = [r.get("stability_churn", 0) for r in churn_results]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot lines with markers
    ax.plot(
        updates,
        baseline_churn,
        marker="o",
        label="Baseline",
        color="#1f77b4",
        linestyle="-",
    )
    ax.plot(
        updates,
        stability_churn,
        marker="s",
        label="Stability-driven",
        color="#2ca02c",
        linestyle="-",
    )

    # Add error bars if standard deviations are available
    baseline_std = [r.get("baseline_churn_std") for r in churn_results]
    stability_std = [r.get("stability_churn_std") for r in churn_results]

    if all(s is not None for s in baseline_std):
        ax.fill_between(
            updates,
            np.array(baseline_churn) - np.array(baseline_std),
            np.array(baseline_churn) + np.array(baseline_std),
            alpha=0.2,
            color="#1f77b4",
        )

    if all(s is not None for s in stability_std):
        ax.fill_between(
            updates,
            np.array(stability_churn) - np.array(stability_std),
            np.array(stability_churn) + np.array(stability_std),
            alpha=0.2,
            color="#2ca02c",
        )

    ax.set_xlabel("Update Index")
    ax.set_ylabel("Neighbor Churn Rate")
    ax.set_title("Neighbor Churn Over Sequential Updates")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_overlap_by_k(
    overlap_results: dict[str, Any],
    k_values: list[int],
    output_path: Path,
) -> None:
    """Plot bar chart of top-k overlap for different k values.

    Args:
        overlap_results: Dictionary containing overlap metrics with keys:
            - 'baseline': dict mapping k -> overlap value
            - 'stability': dict mapping k -> overlap value
        k_values: List of k values to plot (e.g., [10, 50, 100]).
        output_path: Path to save the PDF figure.
    """
    set_paper_style()

    baseline_overlap = overlap_results.get("baseline", {})
    stability_overlap = overlap_results.get("stability", {})

    # Convert k_values to strings for dictionary lookup if needed
    baseline_values = []
    stability_values = []
    for k in k_values:
        baseline_values.append(baseline_overlap.get(k, baseline_overlap.get(str(k), 0)))
        stability_values.append(stability_overlap.get(k, stability_overlap.get(str(k), 0)))

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        stability_values,
        width,
        label="Stability-driven",
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Overlap Ratio")
    ax.set_title("Top-k Neighbor Overlap Before/After Updates")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_chunk_size_distribution(
    chunks_baseline: list[int | float],
    chunks_stability: list[int | float],
    output_path: Path,
) -> None:
    """Plot histogram comparing chunk size distributions.

    Args:
        chunks_baseline: List of chunk sizes (in bytes) for baseline method.
        chunks_stability: List of chunk sizes (in bytes) for stability-driven method.
        output_path: Path to save the PDF figure.
    """
    set_paper_style()

    baseline_sizes = np.array(chunks_baseline) if chunks_baseline else np.array([])
    stability_sizes = np.array(chunks_stability) if chunks_stability else np.array([])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Determine common bins
    all_sizes = np.concatenate([baseline_sizes, stability_sizes]) if len(baseline_sizes) > 0 and len(stability_sizes) > 0 else np.array([256, 4096])
    min_size = max(0, all_sizes.min() - 100)
    max_size = all_sizes.max() + 100
    bins = np.linspace(min_size, max_size, 31)

    # Plot histograms
    if len(baseline_sizes) > 0:
        ax.hist(
            baseline_sizes,
            bins=bins,
            alpha=0.7,
            label=f"Baseline (n={len(baseline_sizes)}, mean={baseline_sizes.mean():.0f})",
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
        )

    if len(stability_sizes) > 0:
        ax.hist(
            stability_sizes,
            bins=bins,
            alpha=0.7,
            label=f"Stability-driven (n={len(stability_sizes)}, mean={stability_sizes.mean():.0f})",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Chunk Size (bytes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Chunk Sizes")
    ax.legend(loc="upper right")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_boundary_sensitivity(
    boundary_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot drift/churn near boundaries vs interior regions.

    Args:
        boundary_metrics: Dictionary containing boundary analysis with keys:
            - 'boundary_drift': list of drift values near chunk boundaries
            - 'interior_drift': list of drift values in chunk interiors
            - 'boundary_churn': list of churn values near boundaries
            - 'interior_churn': list of churn values in interiors
            Optionally includes 'boundary_window_bytes' for context.
        output_path: Path to save the PDF figure.
    """
    set_paper_style()

    boundary_drift = np.array(boundary_metrics.get("boundary_drift", []))
    interior_drift = np.array(boundary_metrics.get("interior_drift", []))
    boundary_churn = np.array(boundary_metrics.get("boundary_churn", []))
    interior_churn = np.array(boundary_metrics.get("interior_churn", []))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Drift comparison
    ax1 = axes[0]

    drift_data = []
    drift_labels = []
    drift_colors = []

    if len(boundary_drift) > 0:
        drift_data.append(boundary_drift)
        drift_labels.append(f"Boundary\n(n={len(boundary_drift)})")
        drift_colors.append("#d62728")

    if len(interior_drift) > 0:
        drift_data.append(interior_drift)
        drift_labels.append(f"Interior\n(n={len(interior_drift)})")
        drift_colors.append("#1f77b4")

    if drift_data:
        bp1 = ax1.boxplot(
            drift_data,
            labels=drift_labels,
            patch_artist=True,
        )
        for patch, color in zip(bp1["boxes"], drift_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.set_ylabel("Embedding Drift (L2 distance)")
    ax1.set_title("Drift: Boundary vs Interior")

    # Right plot: Churn comparison
    ax2 = axes[1]

    churn_data = []
    churn_labels = []
    churn_colors = []

    if len(boundary_churn) > 0:
        churn_data.append(boundary_churn)
        churn_labels.append(f"Boundary\n(n={len(boundary_churn)})")
        churn_colors.append("#d62728")

    if len(interior_churn) > 0:
        churn_data.append(interior_churn)
        churn_labels.append(f"Interior\n(n={len(interior_churn)})")
        churn_colors.append("#1f77b4")

    if churn_data:
        bp2 = ax2.boxplot(
            churn_data,
            labels=churn_labels,
            patch_artist=True,
        )
        for patch, color in zip(bp2["boxes"], churn_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel("Neighbor Churn Rate")
    ax2.set_title("Churn: Boundary vs Interior")

    # Add boundary window info if available
    window_bytes = boundary_metrics.get("boundary_window_bytes")
    if window_bytes:
        fig.suptitle(
            f"Boundary Sensitivity Analysis (window={window_bytes} bytes)",
            y=1.02,
            fontsize=11,
        )

    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
