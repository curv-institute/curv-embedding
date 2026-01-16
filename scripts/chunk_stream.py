#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Streaming chunking CLI.

Chunks streaming data using stability-driven chunking algorithm.

Usage:
    cat input.txt | uv run scripts/chunk_stream.py --output chunks/
    uv run scripts/chunk_stream.py < input.txt --output chunks/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.chunking.streaming import StreamingChunker


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Streaming document chunking")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for chunks",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4096,
        help="Read buffer size in bytes",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for chunks",
    )

    args = parser.parse_args()

    config = load_config(args.config) if args.config.exists() else load_config()

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    chunker = StreamingChunker(config.chunking)

    total_bytes = 0
    chunk_count = 0

    # Read from stdin in chunks
    while True:
        data = sys.stdin.buffer.read(args.buffer_size)
        if not data:
            break

        total_bytes += len(data)

        for chunk in chunker.feed(data):
            if args.output:
                if args.format == "json":
                    chunk_data = {
                        "index": chunk_count,
                        "byte_start": chunk.byte_start,
                        "byte_end": chunk.byte_end,
                        "content": chunk.content.decode("utf-8", errors="replace"),
                        "cut_score": chunk.cut_score,
                    }
                    (args.output / f"chunk_{chunk_count:04d}.json").write_text(
                        json.dumps(chunk_data, indent=2)
                    )
                else:
                    (args.output / f"chunk_{chunk_count:04d}.txt").write_bytes(
                        chunk.content
                    )
            chunk_count += 1

    # Finalize
    for chunk in chunker.finalize():
        if args.output:
            if args.format == "json":
                chunk_data = {
                    "index": chunk_count,
                    "byte_start": chunk.byte_start,
                    "byte_end": chunk.byte_end,
                    "content": chunk.content.decode("utf-8", errors="replace"),
                    "cut_score": chunk.cut_score,
                }
                (args.output / f"chunk_{chunk_count:04d}.json").write_text(
                    json.dumps(chunk_data, indent=2)
                )
            else:
                (args.output / f"chunk_{chunk_count:04d}.txt").write_bytes(
                    chunk.content
                )
        chunk_count += 1

    print(f"Processed {total_bytes} bytes into {chunk_count} chunks", file=sys.stderr)

    if args.output:
        print(f"Wrote chunks to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
