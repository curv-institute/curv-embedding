#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Offline chunking CLI.

Chunks a document using stability-driven chunking algorithm.

Usage:
    uv run scripts/chunk_offline.py input.txt --output chunks/
    uv run scripts/chunk_offline.py input.txt --manifest manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.chunking.offline import chunk_offline
from src.chunking.manifests import generate_manifest


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Offline document chunking")
    parser.add_argument("input", type=Path, help="Input file to chunk")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for chunks",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Output path for manifest JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for chunks",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    config = load_config(args.config) if args.config.exists() else load_config()

    # Read input
    data = args.input.read_bytes()
    doc_id = args.input.stem

    # Chunk
    chunks = chunk_offline(data, config.chunking)

    print(f"Chunked {len(data)} bytes into {len(chunks)} chunks")

    # Output chunks
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for i, chunk in enumerate(chunks):
            if args.format == "json":
                chunk_data = {
                    "index": i,
                    "byte_start": chunk.byte_start,
                    "byte_end": chunk.byte_end,
                    "content": chunk.content.decode("utf-8", errors="replace"),
                    "cut_score": chunk.cut_score,
                }
                (args.output / f"chunk_{i:04d}.json").write_text(
                    json.dumps(chunk_data, indent=2)
                )
            else:
                (args.output / f"chunk_{i:04d}.txt").write_bytes(chunk.content)

        print(f"Wrote chunks to {args.output}")

    # Output manifest
    if args.manifest:
        manifest = generate_manifest(chunks, doc_id, config)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest to {args.manifest}")

    # Print summary
    sizes = [c.byte_end - c.byte_start for c in chunks]
    print(f"\nChunk sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
