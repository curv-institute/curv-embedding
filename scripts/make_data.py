#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Generate synthetic evaluation data for curv-embedding.

This script creates a corpus of synthetic documents with known properties
(boundaries, anchors) for testing chunking and retrieval algorithms.

Usage:
    uv run scripts/make_data.py --seed 42 --num-docs 100
    uv run scripts/make_data.py --domains text,code --min-size 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Config, load_config
from src.data.generator import SyntheticDocument, generate_corpus
from src.data.manifests import (
    generate_anchors_manifest,
    generate_data_manifest,
    generate_query_families,
    query_families_to_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation data for curv-embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=100,
        help="Number of documents to generate",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="text,code,json,logs",
        help="Comma-separated list of domains (text,code,json,logs)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum document size in bytes",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=65536,
        help="Maximum document size in bytes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/data"),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=None,
        help="Number of query families (default: from config or 5)",
    )
    parser.add_argument(
        "--reformulations",
        type=int,
        default=None,
        help="Reformulations per family (default: from config or 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    return parser.parse_args()


def write_document(doc: SyntheticDocument, corpus_dir: Path) -> None:
    """Write a document to the corpus directory."""
    # Determine file extension based on domain
    extensions = {
        "text": ".txt",
        "code": ".py",
        "json": ".json",
        "logs": ".log",
    }
    ext = extensions.get(doc.domain, ".bin")

    filepath = corpus_dir / f"{doc.doc_id}{ext}"
    filepath.write_bytes(doc.content)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config: Config | None = None
    if args.config and args.config.exists():
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded config from: {args.config}")
    else:
        config = Config()  # Use defaults

    # Parse domains
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    valid_domains = {"text", "code", "json", "logs"}
    invalid = set(domains) - valid_domains
    if invalid:
        print(f"Error: Invalid domains: {invalid}", file=sys.stderr)
        print(f"Valid domains: {valid_domains}", file=sys.stderr)
        return 1

    # Get query family parameters
    num_families = args.num_families
    if num_families is None:
        num_families = config.eval.reformulation_families if config else 5

    reformulations = args.reformulations
    if reformulations is None:
        reformulations = config.eval.reformulations_per_family if config else 10

    # Create output directories
    output_dir = args.output_dir.resolve()
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Output directory: {output_dir}")
        print(f"Seed: {args.seed}")
        print(f"Documents: {args.num_docs}")
        print(f"Domains: {domains}")
        print(f"Size range: {args.min_size} - {args.max_size} bytes")
        print(f"Query families: {num_families}")
        print(f"Reformulations per family: {reformulations}")
        print()

    # Generate corpus
    print(f"Generating {args.num_docs} documents...")
    documents = generate_corpus(
        seed=args.seed,
        num_docs=args.num_docs,
        domains=domains,
        size_range=(args.min_size, args.max_size),
    )

    # Write documents to corpus directory
    print(f"Writing documents to {corpus_dir}...")
    for doc in documents:
        write_document(doc, corpus_dir)

    # Generate and write manifest
    print("Generating manifest...")
    manifest = generate_data_manifest(documents, args.seed, config)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    # Generate and write query families
    print("Generating query families...")
    families = generate_query_families(
        documents=documents,
        num_families=num_families,
        reformulations_per_family=reformulations,
        seed=args.seed,
    )
    families_path = output_dir / "query_families.json"
    with families_path.open("w") as f:
        json.dump(query_families_to_dict(families), f, indent=2)

    # Generate and write anchors manifest
    print("Generating anchors manifest...")
    anchors = generate_anchors_manifest(documents)
    anchors_path = output_dir / "anchors.json"
    with anchors_path.open("w") as f:
        json.dump(anchors, f, indent=2)

    # Print summary
    print()
    print("Generation complete!")
    print(f"  Documents: {len(documents)}")
    print(f"  Total size: {manifest['total_bytes']:,} bytes")
    print(f"  Total boundaries: {manifest['summary']['total_boundaries']}")
    print(f"  Total anchors: {manifest['summary']['total_anchors']}")
    print(f"  Query families: {len(families)}")
    print()
    print("Output files:")
    print(f"  {corpus_dir}/")
    print(f"  {manifest_path}")
    print(f"  {families_path}")
    print(f"  {anchors_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
