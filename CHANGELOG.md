# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
