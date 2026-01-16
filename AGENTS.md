# AGENTS.md — Repository Agent Rules

This file contains authoritative rules for **all automated or human agents** working in this repository.

---

## Project Overview

**Repository**: `https://github.com/curv-institute/curv-embedding`

**Purpose**: Stability-driven automatic chunking for vector database embeddings, supporting offline preprocessing and streaming ingestion.

**Hypothesis**: Chunking based on representational stability diagnostics (curvature, stability margin, neighbor disharmony) produces more stable embedding geometry than heuristic chunking.

---

## Authoritative Rules

### 1. Determinism & Reproducibility

All implementations must:
- Be deterministic given a fixed seed
- Record configuration snapshots
- Write manifests for every run
- Isolate outputs per run (`eval/results/<run_name>/`)

Randomness must be:
- Seeded
- Logged
- Reproducible

### 2. Version Control Discipline (jj + git)

Use **Jujutsu (`jj`) as the primary interface** for local version control, with `git` used only for remote interoperability.

**Required workflow**:
- Use small, single-purpose commits
- One logical change per commit
- Commit messages must be imperative and descriptive

**Push discipline (mandatory)**:
- After completing a task: `jj commit` then `jj git push`
- The canonical remote must always reflect the latest completed state
- No local-only commits unless explicitly instructed

**History hygiene**:
- Do not rewrite published history
- Do not squash commits unless explicitly instructed
- Prefer linear, readable history

### 3. Per-Run Output Isolation

Never overwrite results from previous runs. Always write to:
```
eval/results/<run_name>/
  index.faiss
  meta.sqlite
  manifest.json
  metrics.jsonl
  summary.json
  report.md
  figures/
  tables/
  manifests/
```

### 3A. Experiment Naming Convention (Mandatory)

Run names **must** encode the git tag or version to ensure traceability:

**Pattern**: `<tag>-<experiment>-<timestamp>` or `<experiment>_<tag>`

**Examples**:
- `v1.0.0-baseline-20260116` — baseline run on v1.0.0
- `v1.0.0-ablation-nowK-20260116` — ablation study disabling curvature
- `baseline_v1.0.0` — alternative format
- `v1.1.0-streaming-test-20260117` — streaming mode test on v1.1.0

**Rationale**: Makes it obvious which code state generated a result folder. When reviewing `eval/results/`, the version is immediately visible without opening manifests.

**Enforcement**: The `manifest.json` must include both `run_name` and the git tag/commit hash for cross-validation.

### 4. Parallel Subagent Usage

- Decompose tasks into independent file/module ownership whenever possible
- Use as many parallel subagents as needed to reduce wall-clock time
- Only serialize work when changes require tight coordination
- Establish interfaces and contracts first to enable parallel implementation
- Require subagents to produce outputs that can be merged without ambiguity

### 5. Scope Discipline

**Must**:
- Respect frozen components (ULT, HHC, LIL from prior work)
- Treat new work as additive or orthogonal
- Avoid refactoring or "cleanups" unless directly required

**Must not**:
- Introduce new learning objectives unless requested
- Change experimental semantics when asked to do editorial fixes
- Broaden scope beyond the stated hypothesis

### 6. Python Implementation Rules

All Python code must:
- Be compatible with Python >= 3.12
- Use PEP 723 inline script headers for runnable scripts
- Be executable via `uv run`
- Avoid global state where possible
- Log seeds and configuration at runtime

### 7. Metrics & Claims

- Never conflate structural metrics with end-to-end performance
- Claims must follow directly from reported metrics
- Include negative results where present
- State trade-offs explicitly

### 8. AGENT Prompt Archival

Any prompt longer than two lines must be saved as:
```
AGENT/<UNIXTIME>-in.md   # prompt
AGENT/<UNIXTIME>-out.md  # output
```

All AGENT files must be committed and pushed.

---

## Non-Goals

- Do not modify ULT, HHC, or LIL semantics
- Do not introduce learning into chunking
- Do not claim asymptotic ANN speedups
- Do not optimize embedding accuracy benchmarks

---

## Guiding Principle

> **Treat representation as a controlled substrate, not an emergent accident.**
