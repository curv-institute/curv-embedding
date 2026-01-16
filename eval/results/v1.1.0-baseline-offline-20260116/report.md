# Evaluation Report: v1.1.0-baseline-offline-20260116

## Summary

This report summarizes the evaluation results comparing baseline (fixed-size)
chunking against stability-driven chunking for embedding stability.

## Embedding Drift

Drift measures how much chunk embeddings change after document updates.
Lower drift indicates more stable embeddings.

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Count | 0 | 0 |
| Mean | 0.0000 | 0.0000 |
| Std | 0.0000 | 0.0000 |
| P50 | 0.0000 | 0.0000 |
| P90 | 0.0000 | 0.0000 |
| P99 | 0.0000 | 0.0000 |

## Neighbor Churn

Churn measures how much the nearest neighbors of a query change after updates.
Lower churn indicates more stable retrieval results.

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Updates | 0 | 0 |
| Mean | 0.0000 | 0.0000 |
| Std | 0.0000 | 0.0000 |
| P50 | 0.0000 | 0.0000 |
| P90 | 0.0000 | 0.0000 |
| Max | 0.0000 | 0.0000 |

## Top-k Overlap

Overlap measures the fraction of top-k neighbors that remain the same after updates.
Higher overlap indicates more stable retrieval results.

| k | Baseline | Stability-driven |
|---|----------|------------------|
| k=10 | 0.0000 | 0.0000 |
| k=50 | 0.0000 | 0.0000 |
| k=100 | 0.0000 | 0.0000 |

## Chunk Size Distribution

| Metric | Baseline | Stability-driven |
|--------|----------|------------------|
| Count | 0 | 0 |
| Mean (bytes) | 0 | 0 |
| Std | 0 | 0 |
| Min | 0 | 0 |
| Max | 0 | 0 |

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

Include with: `\input{tables/drift_results.tex}`
