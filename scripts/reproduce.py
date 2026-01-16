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
Reproducibility script for curv-embedding experiments.

Validates and reproduces experiment results from manifests.

Usage:
    uv run scripts/reproduce.py --run-name my_experiment
    uv run scripts/reproduce.py --run-name my_experiment --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


def setup_logging(log_level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def verify_manifest(run_dir: Path) -> tuple[bool, list[str]]:
    """Verify all artifacts exist and manifests are valid."""
    issues = []

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append(f"Missing manifest: {manifest_path}")
        return False, issues

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check required fields
    required = ["run_id", "config", "timestamp", "seed", "artifacts"]
    for field in required:
        if field not in manifest:
            issues.append(f"Missing manifest field: {field}")

    # Check artifacts exist
    artifacts = manifest.get("artifacts", {})
    for name, path in artifacts.items():
        if not Path(path).exists():
            issues.append(f"Missing artifact {name}: {path}")

    # Check summary
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        issues.append(f"Missing summary: {summary_path}")

    # Check metrics
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        issues.append(f"Missing metrics: {metrics_path}")
    else:
        # Validate JSONL format
        with open(metrics_path) as f:
            for i, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON at metrics line {i}: {e}")

    return len(issues) == 0, issues


def reproduce_from_manifest(
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict:
    """Reproduce experiment from manifest."""
    logger = logging.getLogger(__name__)

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    config = Config.from_dict(manifest["config"])
    seed = manifest["seed"]

    logger.info(f"Reproducing run: {manifest['run_id']}")
    logger.info(f"Original seed: {seed}")
    logger.info(f"Config hash: {config.config_hash()}")

    if output_dir is None:
        output_dir = run_dir.parent / f"{run_dir.name}_reproduced"

    # Import here to avoid circular dependency
    from scripts.run_experiment import run_experiment

    summary = run_experiment(
        run_name=output_dir.name,
        config=config,
    )

    return summary


def compare_runs(original_dir: Path, reproduced_dir: Path) -> dict:
    """Compare original and reproduced runs."""
    logger = logging.getLogger(__name__)

    comparison = {
        "match": True,
        "differences": [],
    }

    # Load summaries
    with open(original_dir / "summary.json") as f:
        original = json.load(f)
    with open(reproduced_dir / "summary.json") as f:
        reproduced = json.load(f)

    # Compare key metrics
    for method in ["baseline", "stability"]:
        orig_chunks = original.get(method, {}).get("num_chunks", 0)
        repr_chunks = reproduced.get(method, {}).get("num_chunks", 0)

        if orig_chunks != repr_chunks:
            comparison["match"] = False
            comparison["differences"].append({
                "field": f"{method}_num_chunks",
                "original": orig_chunks,
                "reproduced": repr_chunks,
            })

    # Compare config hashes
    if original.get("config_hash") != reproduced.get("config_hash"):
        comparison["match"] = False
        comparison["differences"].append({
            "field": "config_hash",
            "original": original.get("config_hash"),
            "reproduced": reproduced.get("config_hash"),
        })

    if comparison["match"]:
        logger.info("Reproduction successful - results match")
    else:
        logger.warning(f"Reproduction mismatch: {len(comparison['differences'])} differences")
        for diff in comparison["differences"]:
            logger.warning(f"  {diff['field']}: {diff['original']} vs {diff['reproduced']}")

    return comparison


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce experiment from manifest"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the experiment run to reproduce",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for reproduced results",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify manifest without reproducing",
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

    run_dir = args.results_dir / args.run_name

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1

    # Verify manifest
    logger.info(f"Verifying manifest for: {args.run_name}")
    valid, issues = verify_manifest(run_dir)

    if not valid:
        logger.error("Manifest verification failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return 1

    logger.info("Manifest verification passed")

    if args.verify_only:
        return 0

    # Reproduce
    try:
        summary = reproduce_from_manifest(run_dir, args.output_dir)

        # Compare
        reproduced_dir = args.output_dir or (run_dir.parent / f"{run_dir.name}_reproduced")
        comparison = compare_runs(run_dir, reproduced_dir)

        print(json.dumps(comparison, indent=2))
        return 0 if comparison["match"] else 1

    except Exception as e:
        logger.error(f"Reproduction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
