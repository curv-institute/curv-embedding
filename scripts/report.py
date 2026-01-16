#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""
Generate evaluation reports and figures for curv-embedding experiments.

Loads metrics from experiment results and generates:
- PDF figures for drift, churn, overlap, and chunk size analysis
- LaTeX tables for inclusion in papers
- Markdown summary report

Usage:
    uv run scripts/report.py --run-name my_experiment
    uv run scripts/report.py --run-name my_experiment --format latex
    uv run scripts/report.py --run-name my_experiment --results-dir eval/results --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    """Load metrics from JSONL file.

    Args:
        metrics_path: Path to metrics.jsonl file.

    Returns:
        List of metric dictionaries, one per line.
    """
    metrics = []
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found: {metrics_path}", file=sys.stderr)
        return metrics

    with metrics_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    return metrics


def load_summary(summary_path: Path) -> dict[str, Any]:
    """Load summary from JSON file.

    Args:
        summary_path: Path to summary.json file.

    Returns:
        Summary dictionary.
    """
    if not summary_path.exists():
        print(f"Warning: Summary file not found: {summary_path}", file=sys.stderr)
        return {}

    with summary_path.open("r") as f:
        return json.load(f)


def extract_drift_results(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract drift results from metrics and summary.

    Args:
        metrics: List of metric records.
        summary: Summary dictionary.

    Returns:
        Dictionary with baseline and stability drift values.
    """
    drift_results: dict[str, Any] = {
        "baseline": [],
        "stability": [],
    }

    # Extract from metrics (per-chunk drift values)
    for record in metrics:
        if record.get("metric_type") == "drift":
            method = record.get("method", "unknown")
            value = record.get("drift_value", record.get("value"))
            if value is not None:
                if method == "baseline":
                    drift_results["baseline"].append(value)
                elif method in ("stability", "stability_driven"):
                    drift_results["stability"].append(value)

    # Also check summary for aggregated drift data
    if "drift" in summary:
        drift_summary = summary["drift"]
        if "baseline_values" in drift_summary:
            drift_results["baseline"].extend(drift_summary["baseline_values"])
        if "stability_values" in drift_summary:
            drift_results["stability"].extend(drift_summary["stability_values"])

    return drift_results


def extract_churn_results(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract churn results from metrics and summary.

    Args:
        metrics: List of metric records.
        summary: Summary dictionary.

    Returns:
        List of dictionaries with churn per update.
    """
    churn_by_update: dict[int, dict[str, Any]] = {}

    # Extract from metrics
    for record in metrics:
        if record.get("metric_type") == "churn":
            update_idx = record.get("update_idx", 0)
            method = record.get("method", "unknown")
            value = record.get("churn_value", record.get("value"))

            if update_idx not in churn_by_update:
                churn_by_update[update_idx] = {"update_idx": update_idx}

            if method == "baseline":
                churn_by_update[update_idx]["baseline_churn"] = value
            elif method in ("stability", "stability_driven"):
                churn_by_update[update_idx]["stability_churn"] = value

    # Also check summary for churn data
    if "churn" in summary:
        churn_summary = summary["churn"]
        if "per_update" in churn_summary:
            for update_data in churn_summary["per_update"]:
                update_idx = update_data.get("update_idx", len(churn_by_update))
                if update_idx not in churn_by_update:
                    churn_by_update[update_idx] = {"update_idx": update_idx}
                churn_by_update[update_idx].update(update_data)

    # Sort by update index
    return [churn_by_update[k] for k in sorted(churn_by_update.keys())]


def extract_overlap_results(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract overlap results from metrics and summary.

    Args:
        metrics: List of metric records.
        summary: Summary dictionary.

    Returns:
        Dictionary with baseline and stability overlap by k.
    """
    overlap_results: dict[str, Any] = {
        "baseline": {},
        "stability": {},
    }

    # Extract from metrics
    for record in metrics:
        if record.get("metric_type") == "overlap":
            method = record.get("method", "unknown")
            k = record.get("k")
            value = record.get("overlap_value", record.get("value"))

            if k is not None and value is not None:
                if method == "baseline":
                    overlap_results["baseline"][k] = value
                elif method in ("stability", "stability_driven"):
                    overlap_results["stability"][k] = value

    # Also check summary
    if "overlap" in summary:
        overlap_summary = summary["overlap"]
        if "baseline" in overlap_summary:
            overlap_results["baseline"].update(overlap_summary["baseline"])
        if "stability" in overlap_summary:
            overlap_results["stability"].update(overlap_summary["stability"])

    return overlap_results


def extract_chunk_sizes(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Extract chunk size distributions from metrics and summary.

    Args:
        metrics: List of metric records.
        summary: Summary dictionary.

    Returns:
        Tuple of (baseline_sizes, stability_sizes).
    """
    baseline_sizes: list[int] = []
    stability_sizes: list[int] = []

    # Extract from metrics
    for record in metrics:
        if record.get("metric_type") == "chunk_size":
            method = record.get("method", "unknown")
            size = record.get("size_bytes", record.get("size"))

            if size is not None:
                if method == "baseline":
                    baseline_sizes.append(size)
                elif method in ("stability", "stability_driven"):
                    stability_sizes.append(size)

    # Also check summary
    if "chunks" in summary:
        chunk_summary = summary["chunks"]
        if "baseline_sizes" in chunk_summary:
            baseline_sizes.extend(chunk_summary["baseline_sizes"])
        if "stability_sizes" in chunk_summary:
            stability_sizes.extend(chunk_summary["stability_sizes"])

    return baseline_sizes, stability_sizes


def extract_boundary_metrics(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract boundary sensitivity metrics from metrics and summary.

    Args:
        metrics: List of metric records.
        summary: Summary dictionary.

    Returns:
        Dictionary with boundary vs interior drift and churn.
    """
    boundary_metrics: dict[str, Any] = {
        "boundary_drift": [],
        "interior_drift": [],
        "boundary_churn": [],
        "interior_churn": [],
    }

    # Extract from metrics
    for record in metrics:
        if record.get("metric_type") == "boundary_sensitivity":
            location = record.get("location", "unknown")
            drift = record.get("drift_value")
            churn = record.get("churn_value")

            if location == "boundary":
                if drift is not None:
                    boundary_metrics["boundary_drift"].append(drift)
                if churn is not None:
                    boundary_metrics["boundary_churn"].append(churn)
            elif location == "interior":
                if drift is not None:
                    boundary_metrics["interior_drift"].append(drift)
                if churn is not None:
                    boundary_metrics["interior_churn"].append(churn)

    # Also check summary
    if "boundary_sensitivity" in summary:
        bs = summary["boundary_sensitivity"]
        for key in ["boundary_drift", "interior_drift", "boundary_churn", "interior_churn"]:
            if key in bs:
                boundary_metrics[key].extend(bs[key])
        if "boundary_window_bytes" in bs:
            boundary_metrics["boundary_window_bytes"] = bs["boundary_window_bytes"]

    return boundary_metrics


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary with mean, std, min, max, p50, p90, p99.
    """
    import numpy as np

    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "count": 0,
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(arr),
    }


def generate_drift_table_latex(drift_results: dict[str, Any]) -> str:
    """Generate LaTeX table for drift results.

    Args:
        drift_results: Dictionary with baseline and stability drift values.

    Returns:
        LaTeX tabular string.
    """
    baseline_stats = compute_statistics(drift_results.get("baseline", []))
    stability_stats = compute_statistics(drift_results.get("stability", []))

    return f"""\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Method & Count & Mean & Std & P50 & P90 & P99 \\\\
\\midrule
Baseline & {baseline_stats['count']} & {baseline_stats['mean']:.4f} & {baseline_stats['std']:.4f} & {baseline_stats['p50']:.4f} & {baseline_stats['p90']:.4f} & {baseline_stats['p99']:.4f} \\\\
Stability-driven & {stability_stats['count']} & {stability_stats['mean']:.4f} & {stability_stats['std']:.4f} & {stability_stats['p50']:.4f} & {stability_stats['p90']:.4f} & {stability_stats['p99']:.4f} \\\\
\\bottomrule
\\end{{tabular}}"""


def generate_churn_table_latex(churn_results: list[dict[str, Any]]) -> str:
    """Generate LaTeX table for churn results.

    Args:
        churn_results: List of churn results per update.

    Returns:
        LaTeX tabular string.
    """
    baseline_churns = [r.get("baseline_churn", 0) for r in churn_results if r.get("baseline_churn") is not None]
    stability_churns = [r.get("stability_churn", 0) for r in churn_results if r.get("stability_churn") is not None]

    baseline_stats = compute_statistics(baseline_churns)
    stability_stats = compute_statistics(stability_churns)

    return f"""\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Method & Updates & Mean & Std & P50 & P90 & Max \\\\
\\midrule
Baseline & {baseline_stats['count']} & {baseline_stats['mean']:.4f} & {baseline_stats['std']:.4f} & {baseline_stats['p50']:.4f} & {baseline_stats['p90']:.4f} & {baseline_stats['max']:.4f} \\\\
Stability-driven & {stability_stats['count']} & {stability_stats['mean']:.4f} & {stability_stats['std']:.4f} & {stability_stats['p50']:.4f} & {stability_stats['p90']:.4f} & {stability_stats['max']:.4f} \\\\
\\bottomrule
\\end{{tabular}}"""


def generate_overlap_table_latex(
    overlap_results: dict[str, Any],
    k_values: list[int],
) -> str:
    """Generate LaTeX table for overlap results.

    Args:
        overlap_results: Dictionary with baseline and stability overlap by k.
        k_values: List of k values to include.

    Returns:
        LaTeX tabular string.
    """
    # Build header
    header_cols = " & ".join([f"k={k}" for k in k_values])
    header = f"Method & {header_cols}"

    # Build baseline row
    baseline = overlap_results.get("baseline", {})
    baseline_vals = []
    for k in k_values:
        val = baseline.get(k, baseline.get(str(k), 0))
        baseline_vals.append(f"{val:.4f}")
    baseline_row = "Baseline & " + " & ".join(baseline_vals)

    # Build stability row
    stability = overlap_results.get("stability", {})
    stability_vals = []
    for k in k_values:
        val = stability.get(k, stability.get(str(k), 0))
        stability_vals.append(f"{val:.4f}")
    stability_row = "Stability-driven & " + " & ".join(stability_vals)

    col_spec = "l" + "r" * len(k_values)

    return f"""\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header} \\\\
\\midrule
{baseline_row} \\\\
{stability_row} \\\\
\\bottomrule
\\end{{tabular}}"""


def generate_maintenance_table_latex(summary: dict[str, Any]) -> str:
    """Generate LaTeX table for maintenance/overhead results.

    Args:
        summary: Summary dictionary with maintenance metrics.

    Returns:
        LaTeX tabular string.
    """
    maintenance = summary.get("maintenance", {})

    baseline_time = maintenance.get("baseline_update_time_ms", 0)
    stability_time = maintenance.get("stability_update_time_ms", 0)
    baseline_rechunks = maintenance.get("baseline_rechunks", 0)
    stability_rechunks = maintenance.get("stability_rechunks", 0)
    baseline_reembeds = maintenance.get("baseline_reembeddings", 0)
    stability_reembeds = maintenance.get("stability_reembeddings", 0)

    return f"""\\begin{{tabular}}{{lrrr}}
\\toprule
Method & Update Time (ms) & Re-chunks & Re-embeddings \\\\
\\midrule
Baseline & {baseline_time:.2f} & {baseline_rechunks} & {baseline_reembeds} \\\\
Stability-driven & {stability_time:.2f} & {stability_rechunks} & {stability_reembeds} \\\\
\\bottomrule
\\end{{tabular}}"""


def generate_markdown_report(
    run_name: str,
    drift_results: dict[str, Any],
    churn_results: list[dict[str, Any]],
    overlap_results: dict[str, Any],
    chunk_sizes: tuple[list[int], list[int]],
    summary: dict[str, Any],
    k_values: list[int],
) -> str:
    """Generate markdown summary report.

    Args:
        run_name: Name of the experiment run.
        drift_results: Drift analysis results.
        churn_results: Churn analysis results.
        overlap_results: Overlap analysis results.
        chunk_sizes: Tuple of (baseline_sizes, stability_sizes).
        summary: Summary dictionary.
        k_values: List of k values for overlap.

    Returns:
        Markdown string.
    """
    baseline_drift_stats = compute_statistics(drift_results.get("baseline", []))
    stability_drift_stats = compute_statistics(drift_results.get("stability", []))

    baseline_churns = [r.get("baseline_churn", 0) for r in churn_results if r.get("baseline_churn") is not None]
    stability_churns = [r.get("stability_churn", 0) for r in churn_results if r.get("stability_churn") is not None]
    baseline_churn_stats = compute_statistics(baseline_churns)
    stability_churn_stats = compute_statistics(stability_churns)

    baseline_sizes, stability_sizes = chunk_sizes
    baseline_size_stats = compute_statistics([float(x) for x in baseline_sizes])
    stability_size_stats = compute_statistics([float(x) for x in stability_sizes])

    # Calculate improvement percentages
    drift_improvement = ""
    if baseline_drift_stats["mean"] > 0:
        pct = (1 - stability_drift_stats["mean"] / baseline_drift_stats["mean"]) * 100
        drift_improvement = f" ({pct:+.1f}%)"

    churn_improvement = ""
    if baseline_churn_stats["mean"] > 0:
        pct = (1 - stability_churn_stats["mean"] / baseline_churn_stats["mean"]) * 100
        churn_improvement = f" ({pct:+.1f}%)"

    # Build overlap table
    overlap_rows = []
    for k in k_values:
        b_val = overlap_results.get("baseline", {}).get(k, overlap_results.get("baseline", {}).get(str(k), 0))
        s_val = overlap_results.get("stability", {}).get(k, overlap_results.get("stability", {}).get(str(k), 0))
        overlap_rows.append(f"| k={k} | {b_val:.4f} | {s_val:.4f} |")

    overlap_table = "\n".join(overlap_rows)

    report = f"""# Evaluation Report: {run_name}

## Summary

This report summarizes the evaluation results comparing baseline (fixed-size)
chunking against stability-driven chunking for embedding stability.

## Embedding Drift

Drift measures how much chunk embeddings change after document updates.
Lower drift indicates more stable embeddings.

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Count | {baseline_drift_stats['count']} | {stability_drift_stats['count']} |
| Mean | {baseline_drift_stats['mean']:.4f} | {stability_drift_stats['mean']:.4f}{drift_improvement} |
| Std | {baseline_drift_stats['std']:.4f} | {stability_drift_stats['std']:.4f} |
| P50 | {baseline_drift_stats['p50']:.4f} | {stability_drift_stats['p50']:.4f} |
| P90 | {baseline_drift_stats['p90']:.4f} | {stability_drift_stats['p90']:.4f} |
| P99 | {baseline_drift_stats['p99']:.4f} | {stability_drift_stats['p99']:.4f} |

## Neighbor Churn

Churn measures how much the nearest neighbors of a query change after updates.
Lower churn indicates more stable retrieval results.

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Updates | {baseline_churn_stats['count']} | {stability_churn_stats['count']} |
| Mean | {baseline_churn_stats['mean']:.4f} | {stability_churn_stats['mean']:.4f}{churn_improvement} |
| Std | {baseline_churn_stats['std']:.4f} | {stability_churn_stats['std']:.4f} |
| P50 | {baseline_churn_stats['p50']:.4f} | {stability_churn_stats['p50']:.4f} |
| P90 | {baseline_churn_stats['p90']:.4f} | {stability_churn_stats['p90']:.4f} |
| Max | {baseline_churn_stats['max']:.4f} | {stability_churn_stats['max']:.4f} |

## Top-k Overlap

Overlap measures the fraction of top-k neighbors that remain the same after updates.
Higher overlap indicates more stable retrieval results.

| k | Baseline | Stability-driven |
|---|----------|------------------|
{overlap_table}

## Chunk Size Distribution

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Count | {baseline_size_stats['count']} | {stability_size_stats['count']} |
| Mean (bytes) | {baseline_size_stats['mean']:.0f} | {stability_size_stats['mean']:.0f} |
| Std | {baseline_size_stats['std']:.0f} | {stability_size_stats['std']:.0f} |
| Min | {baseline_size_stats['min']:.0f} | {stability_size_stats['min']:.0f} |
| Max | {baseline_size_stats['max']:.0f} | {stability_size_stats['max']:.0f} |

## Figures

- [Drift Distribution](figures/drift_distribution.pdf)
- [Churn Over Updates](figures/churn_over_updates.pdf)
- [Overlap by k](figures/overlap_by_k.pdf)
- [Chunk Sizes](figures/chunk_sizes.pdf)

## LaTeX Tables

Tables are available in the `tables/` directory for inclusion in papers:
- `tables/drift_results.tex`
- `tables/churn_results.tex`
- `tables/overlap_results.tex`
- `tables/maintenance_results.tex`

Include with: `\\input{{tables/drift_results.tex}}`
"""

    return report


def main() -> int:
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports and figures for curv-embedding experiments."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the experiment run",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing experiment results (default: eval/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: same as results)",
    )
    parser.add_argument(
        "--format",
        choices=["latex", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Determine paths
    run_dir = args.results_dir / args.run_name
    output_dir = args.output_dir if args.output_dir else run_dir

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    # Load data
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"

    print(f"Loading metrics from {metrics_path}")
    metrics = load_metrics(metrics_path)

    print(f"Loading summary from {summary_path}")
    summary = load_summary(summary_path)

    # Extract results
    drift_results = extract_drift_results(metrics, summary)
    churn_results = extract_churn_results(metrics, summary)
    overlap_results = extract_overlap_results(metrics, summary)
    chunk_sizes = extract_chunk_sizes(metrics, summary)
    boundary_metrics = extract_boundary_metrics(metrics, summary)

    # Get k values from summary or use defaults
    k_values = summary.get("config", {}).get("eval", {}).get("top_k", [10, 50, 100])

    # Create output directories
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Import plotting module (lazy import to handle missing matplotlib gracefully)
    # We add the src directory to the path to import the plots module
    src_dir = Path(__file__).parent.parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    try:
        from eval.plots import (
            plot_boundary_sensitivity,
            plot_chunk_size_distribution,
            plot_churn_over_updates,
            plot_drift_distribution,
            plot_overlap_by_k,
        )

        # Generate figures
        print("Generating figures...")

        print(f"  - {figures_dir / 'drift_distribution.pdf'}")
        plot_drift_distribution(drift_results, figures_dir / "drift_distribution.pdf")

        print(f"  - {figures_dir / 'churn_over_updates.pdf'}")
        plot_churn_over_updates(churn_results, figures_dir / "churn_over_updates.pdf")

        print(f"  - {figures_dir / 'overlap_by_k.pdf'}")
        plot_overlap_by_k(overlap_results, k_values, figures_dir / "overlap_by_k.pdf")

        print(f"  - {figures_dir / 'chunk_sizes.pdf'}")
        baseline_sizes, stability_sizes = chunk_sizes
        plot_chunk_size_distribution(baseline_sizes, stability_sizes, figures_dir / "chunk_sizes.pdf")

        # Only generate boundary plot if we have data
        if any(boundary_metrics.get(k) for k in ["boundary_drift", "interior_drift", "boundary_churn", "interior_churn"]):
            print(f"  - {figures_dir / 'boundary_sensitivity.pdf'}")
            plot_boundary_sensitivity(boundary_metrics, figures_dir / "boundary_sensitivity.pdf")

    except ImportError as e:
        print(f"Warning: Could not import plotting module: {e}", file=sys.stderr)
        print("Skipping figure generation.", file=sys.stderr)

    # Generate tables (LaTeX)
    if args.format in ("latex", "both"):
        print("Generating LaTeX tables...")

        drift_table = generate_drift_table_latex(drift_results)
        drift_table_path = tables_dir / "drift_results.tex"
        print(f"  - {drift_table_path}")
        with drift_table_path.open("w") as f:
            f.write(drift_table)

        churn_table = generate_churn_table_latex(churn_results)
        churn_table_path = tables_dir / "churn_results.tex"
        print(f"  - {churn_table_path}")
        with churn_table_path.open("w") as f:
            f.write(churn_table)

        overlap_table = generate_overlap_table_latex(overlap_results, k_values)
        overlap_table_path = tables_dir / "overlap_results.tex"
        print(f"  - {overlap_table_path}")
        with overlap_table_path.open("w") as f:
            f.write(overlap_table)

        maintenance_table = generate_maintenance_table_latex(summary)
        maintenance_table_path = tables_dir / "maintenance_results.tex"
        print(f"  - {maintenance_table_path}")
        with maintenance_table_path.open("w") as f:
            f.write(maintenance_table)

    # Generate markdown report
    if args.format in ("markdown", "both"):
        print("Generating markdown report...")

        report = generate_markdown_report(
            args.run_name,
            drift_results,
            churn_results,
            overlap_results,
            chunk_sizes,
            summary,
            k_values,
        )
        report_path = output_dir / "report.md"
        print(f"  - {report_path}")
        with report_path.open("w") as f:
            f.write(report)

    print(f"\nReport generation complete. Output in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
