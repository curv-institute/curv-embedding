# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-01-16

### Added
- Representational reranking over ANN candidates using stability diagnostics
- `src/eval/reranker.py` — RepresentationalReranker module
- `scripts/run_rerank_experiment.py` — Three-mode rerank evaluation
- RerankConfig with alpha, beta, gamma, delta weights and K0/k parameters
- Reranking formula: score = alpha*sim - beta*strain - gamma*boundary - delta*risk

### Results
- Reformulation Overlap: ANN 88.7%, Random 20.1%, Repr 90.7%
- Repr improves overlap by 2.3% over ANN baseline
- P10 overlap improved from 0.700 to 0.800 (+14.3% worst-case improvement)
- Disagreement rate: 87.8% (active reranking, local adjustments)

### Technical Notes
- **v1.6.0 Context:** Count-based mutation metrics mischaracterized hybrid chunking by treating all chunks as equal cost. Byte-weighted metrics showed that hybrid chunking reduces effective mutation cost by 54% relative to baseline, resolving the apparent negative result and establishing hybrid chunking as the correct base policy for dynamic vector stores.
- First cycle that changes retrieval behavior, not chunking or embedding
- AGENT prompt used: `AGENT/1768597042-in.md`

### Paper Artifacts
- `paper/tables/rerank_stability_v2.0.0.tex`
- `paper/figures/rerank_overlap_v2.0.0.pdf`
- `paper/figures/rerank_boundary_v2.0.0.pdf`

## [1.6.0] - 2026-01-16

### Added
- Cost-weighted mutation metrics (byte-weighted re-embed fraction)
- `scripts/compute_cost_weighted_metrics.py` for post-hoc re-analysis
- Localization efficiency metric for edit window containment

### Results
- Byte-weighted re-embed: Baseline 40.9%, Stability 42.6%, Hybrid 31.5%
- Hybrid achieves 54% lower mutation cost than baseline under byte-weighted metrics
- Ranking reversal: Hybrid now best (was worst under count-based)

### Technical Notes
- **v1.5.0 Takeaway:** Hybrid chunking increased raw re-embed fraction because localized micro-chunking converts few large invalidations into many small ones. Raw chunk-count metrics therefore overstate mutation cost. Localization efficiency (80.7%) indicates that hybrid chunking confines change impact, motivating cost-weighted mutation metrics.
- Measurement-only cycle: no changes to chunking, embedding, or FAISS logic
- AGENT prompt used: `AGENT/1768595902-in.md`

### Paper Artifacts
- `paper/tables/temporal_cost_weighted_v1.5.0.tex`
- `paper/figures/temporal_cost_weighted_v1.5.0.pdf`
- `paper/figures/cost_localization_curves_v1.5.0.pdf`

## [1.5.0] - 2026-01-16

### Added
- Hybrid chunking: stability base + local overlapping micro-chunks in edit windows
- `scripts/run_hybrid_experiment.py` for three-way temporal comparison
- HybridConfig with micro_chunk_bytes, micro_overlap_bytes, guard_band_bytes
- Localization efficiency metric (re-embeds confined to edit windows)
- Edit window tracking with parent_chunk_id and edit_window_id

### Results
- Baseline: 60.7% re-embed, 0.428 top-k overlap, 92 chunks
- Stability: 63.3% re-embed, 0.318 top-k overlap, 55 chunks
- Hybrid: 69.7% re-embed, 0.347 top-k overlap, 67 chunks (52.2% micro)
- Localization Efficiency: 80.7% of re-embeds confined to edit windows

### Technical Notes
- Hypothesis partially supported: hybrid improves churn over stability (+9.1%)
- Trade-off: micro-chunking localizes changes but increases total chunk count
- v1.4.0 takeaway incorporated: coherence vs tolerance are competing objectives
- AGENT prompt used: `AGENT/1768595075-in.md`

### Paper Artifacts
- `paper/tables/temporal_hybrid_v1.4.0-20260116.tex`
- `paper/figures/temporal_hybrid_v1.4.0-20260116.pdf`
- `paper/figures/temporal_localization_v1.4.0-20260116.pdf`

## [1.4.0] - 2026-01-16

### Added
- Temporal stability evaluation under incremental updates
- `scripts/run_temporal_experiment.py` for temporal experiments
- Embedding drift, ANN neighbor churn, and maintenance cost metrics
- Boundary-localized effect analysis with explicit commit horizon
- Update simulation: append-only, local edits, boundary stress scenarios

### Results
- Embedding Drift: 0.0 for both methods (content-matched chunks don't drift)
- ANN Neighbor Churn: Baseline 53.1% overlap vs Stability 32.2%
- Maintenance Cost: Baseline 52.5% re-embed vs Stability 63.4%

### Paper Artifacts
- `paper/tables/temporal_stability_v1.3.0-20260116.tex`
- `paper/figures/temporal_drift_v1.3.0-20260116.pdf`
- `paper/figures/temporal_churn_v1.3.0-20260116.pdf`
- `paper/figures/temporal_maintenance_v1.3.0-20260116.pdf`
- `paper/figures/temporal_boundary_v1.3.0-20260116.pdf`

### Technical Notes
- Negative result: baseline showed better temporal stability than stability-driven
- Trade-off identified: larger chunks are more vulnerable to any edit
- Fixed-size overlap provides different form of update resilience
- AGENT prompt used: `AGENT/1768593339-in.md`

## [1.3.0] - 2026-01-16

### Added
- Streaming chunking parity evaluation (offline vs streaming mode comparison)
- `scripts/run_parity_experiment.py` for automated parity analysis
- Boundary overlap metrics at multiple tolerances (0, 16, 64, 256 bytes)
- Wasserstein-1 distance and KS statistic for size distribution comparison
- `%within_10pct_target` alignment metric for both modes

### Results
- Boundary overlap: 97.5% at ±256 bytes tolerance
- Streaming produces 32.6% more chunks (752 vs 567)
- Wasserstein-1 distance: 719.9 bytes between size distributions

### Paper Artifacts
- `paper/tables/chunk_parity_v1.2.0-parity-20260116.tex`
- `paper/figures/chunk_parity_sizes_v1.2.0-parity-20260116.pdf`
- `paper/figures/chunk_parity_overlap_v1.2.0-parity-20260116.pdf`

### Technical Notes
- Hypothesis confirmed: streaming approximates offline within commit horizon
- AGENT prompt used: `AGENT/1768592643-in.md`

## [1.2.0] - 2026-01-16

### Added
- Run v1.1.0 experiment matrix: baseline vs stability-driven chunking (offline)
- Generated evaluation data: 50 documents, 1.66 MB, 5556 boundaries
- Paper-ready artifacts: `paper/tables/experiment_summary_v1.1.0.tex`
- Chunk size comparison figure: `paper/figures/chunk_sizes_v1.1.0.pdf`

### Results
- Stability-driven chunking produces 34.6% fewer chunks (567 vs 867)
- Mean chunk size increased 51.2% (2,990 vs 1,978 bytes)
- Chunks better aligned with target size (2,048 bytes)

### Fixed
- `run_experiment.py`: Fixed manifest key `doc_ids` (was `document_ids`)

### Technical Notes
- Streaming mode comparison deferred (requires script enhancement)
- AGENT prompt used: `AGENT/1768591673-in.md`

## [1.1.0] - 2026-01-16

### Added
- Run naming convention with git tag/commit traceability in manifests
- Validate run_name format (warning-only) in `run_experiment.py`
- Section 3A in AGENTS.md documenting mandatory naming pattern

### Technical Notes
- AGENT prompt used: `AGENT/1768591313-in.md`

## [1.0.0] - 2026-01-16

### Added
- Initial stability-driven chunking baseline
- Cut-score algorithm with configurable weights and thresholds
- Offline chunking with local maxima boundary selection
- Streaming chunking with commit horizon and soft/hard triggers
- FAISS + SQLite vector store with WAL mode and audit logging
- Evaluation metrics: drift, churn, overlap, maintenance cost
- Synthetic data generation for text, code, JSON, and logs
- Query family generation for reformulation testing
- Paper-quality plotting and LaTeX table generation
- PEP 723 CLI scripts for all operations
- Explicit diagnostic mode labeling (`diagnostics.mode = "proxy_entropy"`)

### Technical Notes
- Proxy diagnostics only: K=byte_entropy, S=inverse_variance, B=newlines
- No HHC/LIL integration in v1.0.0 baseline
- AGENT prompts archived: `AGENT/1768588892-in.md`, `AGENT/1768590839-in.md`

## [0.1.0] - 2026-01-16

### Added
- Initial repository structure
- AGENTS.md and CLAUDE.md with Canonical Prompt Contract
- Repository scaffolding for stability-driven chunking experiments
