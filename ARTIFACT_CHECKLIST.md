# Artifact Checklist

This checklist tracks reproducibility requirements for paper submission.

## Code Artifacts

- [ ] All source code in `src/` is complete and documented
- [ ] All scripts in `scripts/` are executable via `uv run`
- [ ] PEP 723 inline dependencies declared in all scripts
- [ ] `configs/default.toml` contains all experiment parameters
- [ ] Tests pass: `uv run scripts/test_all.py`

## Data Artifacts

- [ ] Synthetic data generation is deterministic given seed
- [ ] `scripts/make_data.py` produces identical output across runs
- [ ] Query families and planted anchors documented in manifests

## Experiment Artifacts

- [ ] Each run isolated under `eval/results/<run_name>/`
- [ ] Per-run manifests include:
  - [ ] `manifest.json` with config snapshot
  - [ ] `metrics.jsonl` with raw measurements
  - [ ] `summary.json` with aggregated statistics
- [ ] FAISS indices (`index.faiss`) and SQLite DBs (`meta.sqlite`) preserved
- [ ] All seeds logged in manifests

## Paper Artifacts

- [ ] Tables generated via `scripts/report.py`
- [ ] Figures generated via `src/eval/plots.py`
- [ ] No hand-edited numbers in `paper/tables/` or `paper/figures/`
- [ ] All figures/tables referenced in `paper/main.tex`

## Reproducibility

- [ ] `scripts/reproduce.py` regenerates all results from manifests
- [ ] Fresh clone + `uv run scripts/reproduce.py` succeeds
- [ ] Numeric results match within floating-point tolerance

## Version Control

- [ ] All AGENT prompts archived in `AGENT/`
- [ ] Clean commit history (no WIP commits)
- [ ] VERSION, CITATION.cff, CHANGELOG.md up to date
