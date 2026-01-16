"""
Configuration loader for curv-embedding.

Provides typed configuration access with validation and manifest generation.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChunkingConfig:
    """Chunking algorithm parameters."""

    min_bytes: int = 256
    max_bytes: int = 4096
    overlap_bytes: int = 64
    commit_horizon_bytes: int = 1024
    L_target_bytes: int = 2048

    # Weights
    wK: float = 1.0
    wD: float = 0.8
    wS: float = 0.6
    wB: float = 2.0
    wL: float = 0.5

    # Thresholds
    k0: float = 0.5
    d0: float = 0.5
    s0: float = 0.5

    # Feature toggles
    use_lil_boundaries: bool = True
    use_curvature: bool = True
    use_disharmony: bool = False
    use_stability_margin: bool = True

    # Streaming
    soft_trigger_threshold: float = 1.5
    soft_trigger_sustain_steps: int = 3


@dataclass
class HybridConfig:
    """Hybrid chunking parameters.

    Combines stability-driven base chunks with localized overlapping
    micro-chunks in edit windows for improved mutation tolerance.
    """

    enabled: bool = True
    micro_chunk_bytes: int = 768
    micro_overlap_bytes: int = 96
    guard_band_bytes: int = 256


@dataclass
class EmbeddingConfig:
    """Embedding model parameters."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = "v2.2.2"
    embedding_dim: int = 384
    batch_size: int = 32
    normalize: bool = True


@dataclass
class StorageConfig:
    """Storage backend parameters."""

    sqlite_journal_mode: str = "WAL"
    sqlite_synchronous: str = "NORMAL"
    sqlite_foreign_keys: bool = True
    faiss_index_type: str = "Flat"
    faiss_metric: str = "L2"


@dataclass
class EvalConfig:
    """Evaluation parameters."""

    top_k: list[int] = field(default_factory=lambda: [10, 50, 100])
    drift_content_match: str = "sha256"
    probe_sample_size: int = 1000
    boundary_window_bytes: int = 256
    reformulation_families: int = 5
    reformulations_per_family: int = 10


@dataclass
class BaselineConfig:
    """Baseline chunking parameters for comparison."""

    method: str = "fixed"
    fixed_chunk_bytes: int = 2048
    fixed_overlap_bytes: int = 64


@dataclass
class DiagnosticsConfig:
    """Diagnostic signal mode configuration.

    This explicitly labels which diagnostic signals are used for chunking.
    v1.0.0 uses proxy diagnostics (entropy/variance) rather than full HHC/LIL.
    """

    mode: str = "proxy_entropy"
    description: str = "v1.0.0 baseline uses proxy diagnostics: K=byte_entropy, S=inverse_variance, B=newlines"


@dataclass
class GeneralConfig:
    """General experiment parameters."""

    seed: int = 42
    output_dir: str = "eval/results"
    log_level: str = "INFO"


@dataclass
class Config:
    """Complete experiment configuration."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load configuration from TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        return cls(
            general=GeneralConfig(**data.get("general", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            hybrid=HybridConfig(**data.get("hybrid", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            storage=StorageConfig(**data.get("storage", {})),
            eval=EvalConfig(**data.get("eval", {})),
            baseline=BaselineConfig(**data.get("baseline", {})),
            diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "general": self.general.__dict__,
            "chunking": self.chunking.__dict__,
            "hybrid": self.hybrid.__dict__,
            "embedding": self.embedding.__dict__,
            "storage": self.storage.__dict__,
            "eval": self.eval.__dict__,
            "baseline": self.baseline.__dict__,
            "diagnostics": self.diagnostics.__dict__,
        }

    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_manifest(self) -> dict[str, Any]:
        """Generate manifest entry for this configuration."""
        return {
            "config": self.to_dict(),
            "config_hash": self.config_hash(),
        }


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from file or return defaults."""
    if path is None:
        return Config()
    return Config.from_toml(path)
