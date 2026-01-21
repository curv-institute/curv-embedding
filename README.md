# Stability-Driven Chunking and Representational Reranking for Dynamic Vector Databases

This repository contains the code, data generation scripts, and experiment manifests for the paper:

**Stability-Driven Chunking and Representational Reranking for Dynamic Vector Databases**
J. W. Miller, CURV Institute
January 2026

**Publication:** https://curv.institute/publications/curv-embedding/

## Abstract

Vector databases are increasingly used in dynamic settings where documents are incrementally ingested, edited, and queried over time. In such regimes, heuristic chunking and similarity-based retrieval can lead to unstable embeddings, excessive re-indexing, and inconsistent retrieval behavior. In this work, we present a staged empirical study of stability-driven chunking and representational reranking for vector databases. We show that chunking based on representational stability produces fewer, more coherent chunks and supports streaming ingestion with bounded deviation from offline segmentation. We report a negative result demonstrating that stability-driven chunking is more sensitive to localized edits under naive, count-based mutation metrics. We then show that this apparent failure is a measurement artifact: when mutation cost is measured in bytes rather than chunk counts, a hybrid chunking policy that combines stability-driven base chunks with localized overlapping micro-chunks achieves the lowest effective update cost. Finally, we demonstrate that representational reranking improves retrieval robustness in the tails, increasing worst-case reformulation stability without affecting mutation cost or determinism.

## Requirements

- Python >= 3.12
- Dependencies declared via PEP 723 inline headers
- Executable via `uv run`

## Reproducing Results

```bash
git clone https://github.com/curv-institute/curv-embedding
cd curv-embedding
uv run scripts/reproduce.py --run-name <run_name>
```

## Manifest Format

Each run produces:

- `manifest.json`: Configuration snapshot and seeds
- `metrics.jsonl`: Raw metric measurements
- `summary.json`: Aggregated statistics
- `meta.sqlite`: Full chunk and event audit log

## License

See LICENSE file for details.
