# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
