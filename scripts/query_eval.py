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
Query evaluation CLI.

Evaluates retrieval quality using query families against a built index.

Usage:
    uv run scripts/query_eval.py index/ --queries queries.json --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.embedding.model import EmbeddingModel
from src.storage.sqlite_store import SQLiteStore
from src.storage.faiss_index import FAISSIndex
from src.eval.overlap import compute_overlap_stats, compute_hit_rate
from src.eval.churn import compute_topk_overlap


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate query retrieval")
    parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing index.faiss and meta.sqlite",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Query families JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10, 50, 100],
        help="Top-k values to evaluate",
    )

    args = parser.parse_args()

    config = load_config(args.config) if args.config.exists() else load_config()

    # Load index
    index_path = args.index_dir / "index.faiss"
    db_path = args.index_dir / "meta.sqlite"

    if not index_path.exists():
        print(f"Error: Index not found: {index_path}", file=sys.stderr)
        return 1

    print("Loading index...")
    faiss_index = FAISSIndex(config.embedding.embedding_dim, config.storage)
    faiss_index.load(index_path)

    print("Loading database...")
    store = SQLiteStore(db_path, config.storage)

    print("Loading embedding model...")
    model = EmbeddingModel(config.embedding)

    # Load queries
    with open(args.queries) as f:
        query_data = json.load(f)

    # Handle different query formats
    if isinstance(query_data, list):
        # List of query families
        query_families = query_data
    elif "families" in query_data:
        query_families = query_data["families"]
    else:
        query_families = [query_data]

    print(f"Evaluating {len(query_families)} query families...")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_dir": str(args.index_dir),
        "k_values": args.k,
        "families": [],
    }

    # Build chunk ID to FAISS ID mapping
    all_chunks = store.get_all_chunks()
    chunk_to_faiss = {c.chunk_id: c.faiss_id for c in all_chunks}
    faiss_to_chunk = {c.faiss_id: c.chunk_id for c in all_chunks}

    for family in query_families:
        family_id = family.get("family_id", "unknown")
        queries = family.get("queries", [])
        expected_anchors = set(family.get("target_anchor_ids", []))

        print(f"\n  Family: {family_id} ({len(queries)} queries)")

        family_results = {
            "family_id": family_id,
            "num_queries": len(queries),
            "expected_anchors": list(expected_anchors),
            "per_query": [],
            "aggregate": {},
        }

        all_retrieved = []

        for query_text in queries:
            # Embed query
            query_embedding = model.embed_single(query_text)

            query_results = {"query": query_text, "results_by_k": {}}

            for k in args.k:
                # Search
                distances, faiss_ids = faiss_index.search(
                    query_embedding.reshape(1, -1), k
                )

                # Map to chunk IDs
                retrieved_chunks = []
                for fid in faiss_ids[0]:
                    if fid >= 0 and fid in faiss_to_chunk:
                        retrieved_chunks.append(faiss_to_chunk[fid])

                # Compute hit rate if we have expected results
                hit_rate = None
                if expected_anchors:
                    hit_rate = compute_hit_rate(
                        retrieved_chunks,
                        expected_anchors,
                        k,
                    )

                query_results["results_by_k"][k] = {
                    "retrieved": retrieved_chunks[:k],
                    "hit_rate": hit_rate,
                }

            family_results["per_query"].append(query_results)
            all_retrieved.append(query_results)

        # Compute reformulation stability (overlap between queries in family)
        for k in args.k:
            if len(all_retrieved) > 1:
                overlaps = []
                for i in range(len(all_retrieved)):
                    for j in range(i + 1, len(all_retrieved)):
                        list_i = all_retrieved[i]["results_by_k"][k]["retrieved"]
                        list_j = all_retrieved[j]["results_by_k"][k]["retrieved"]
                        overlap = compute_topk_overlap(list_i, list_j, k)
                        overlaps.append(overlap)

                mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
            else:
                mean_overlap = 1.0

            # Mean hit rate
            hit_rates = [
                q["results_by_k"][k]["hit_rate"]
                for q in all_retrieved
                if q["results_by_k"][k]["hit_rate"] is not None
            ]
            mean_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else None

            family_results["aggregate"][f"k={k}"] = {
                "mean_reformulation_overlap": mean_overlap,
                "mean_hit_rate": mean_hit_rate,
            }

        results["families"].append(family_results)

    # Aggregate across all families
    results["summary"] = {}
    for k in args.k:
        all_overlaps = []
        all_hit_rates = []

        for fam in results["families"]:
            agg = fam["aggregate"].get(f"k={k}", {})
            if agg.get("mean_reformulation_overlap") is not None:
                all_overlaps.append(agg["mean_reformulation_overlap"])
            if agg.get("mean_hit_rate") is not None:
                all_hit_rates.append(agg["mean_hit_rate"])

        results["summary"][f"k={k}"] = {
            "mean_reformulation_overlap": sum(all_overlaps) / len(all_overlaps) if all_overlaps else None,
            "mean_hit_rate": sum(all_hit_rates) / len(all_hit_rates) if all_hit_rates else None,
        }

    store.close()

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print(json.dumps(results["summary"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
