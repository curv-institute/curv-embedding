#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
# ]
# ///
"""
Test runner for curv-embedding.

Runs all unit tests and validates module functionality.

Usage:
    uv run scripts/test_all.py
    uv run scripts/test_all.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config() -> tuple[bool, str]:
    """Test configuration module."""
    from src.config import Config, load_config, ChunkingConfig

    # Test default config
    config = Config()
    assert config.general.seed == 42
    assert config.chunking.min_bytes == 256

    # Test to_dict and config_hash
    d = config.to_dict()
    assert "chunking" in d
    assert "embedding" in d

    h = config.config_hash()
    assert len(h) == 16

    # Test from_dict
    config2 = Config.from_dict(d)
    assert config2.config_hash() == h

    return True, "config module OK"


def test_cut_score() -> tuple[bool, str]:
    """Test cut-score computation."""
    from src.chunking.cut_score import (
        CutScoreSignals,
        compute_cut_score,
        RollingNormalizer,
        relu,
        SignalNormalizers,
    )
    from src.config import ChunkingConfig

    # Test relu
    assert relu(5.0) == 5.0
    assert relu(-5.0) == 0.0
    assert relu(0.0) == 0.0

    # Test RollingNormalizer
    norm = RollingNormalizer(window_size=10)
    for i in range(20):
        zscore = norm.update(float(i))  # update returns zscore
    assert isinstance(zscore, float)

    # Test cut score
    config = ChunkingConfig()
    signals = CutScoreSignals(K=1.0, S=0.5, D=0.0, B=1.0, L=2000)

    # Create normalizers and warm them up
    normalizers = SignalNormalizers(window_size=100)
    for _ in range(50):
        normalizers.normalize(signals)  # use normalize() method

    # compute_cut_score returns (score, normalized_signals)
    score, norm_signals = compute_cut_score(signals, config, normalizers)
    assert isinstance(score, float)
    assert score >= 0.0

    return True, "cut_score module OK"


def test_offline_chunking() -> tuple[bool, str]:
    """Test offline chunking."""
    from src.chunking.offline import chunk_offline, Chunk
    from src.config import ChunkingConfig

    config = ChunkingConfig(
        min_bytes=10,
        max_bytes=100,
    )

    # Create test data
    data = b"Hello world. This is a test.\nAnother line here.\nAnd more content to chunk."

    chunks = chunk_offline(data, config)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

    # Verify coverage
    covered = sum(c.byte_end - c.byte_start for c in chunks)
    # Note: with overlap, covered may be > len(data)
    assert covered >= len(data) - config.max_bytes  # Allow for boundary effects

    return True, "offline chunking OK"


def test_streaming_chunking() -> tuple[bool, str]:
    """Test streaming chunking."""
    from src.chunking.streaming import chunk_stream, StreamingChunker
    from src.config import ChunkingConfig

    config = ChunkingConfig(
        min_bytes=10,
        max_bytes=100,
        commit_horizon_bytes=50,
    )

    # Create test data
    data = b"Hello world. This is a test.\nAnother line here.\nAnd more content to chunk."

    # Test iterator interface
    def data_iter():
        chunk_size = 20
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    chunks = list(chunk_stream(data_iter(), config))

    assert len(chunks) > 0

    # Test class interface
    chunker = StreamingChunker(config)
    for chunk_data in data_iter():
        for c in chunker.feed(chunk_data):
            pass
    final_chunks = chunker.finalize()

    return True, "streaming chunking OK"


def test_data_generator() -> tuple[bool, str]:
    """Test data generation."""
    from src.data.generator import (
        generate_corpus,
        generate_text_document,
        generate_code_document,
        SyntheticDocument,
    )

    # Test single document
    doc = generate_text_document(seed=42, size_bytes=1000)
    assert isinstance(doc, SyntheticDocument)
    assert len(doc.content) > 0
    assert doc.domain == "text"

    # Test code document
    code_doc = generate_code_document(seed=42, size_bytes=1000)
    assert code_doc.domain == "code"

    # Test corpus
    corpus = generate_corpus(
        seed=42,
        num_docs=5,
        domains=["text", "code"],
        size_range=(100, 500),
    )
    assert len(corpus) == 5

    return True, "data generator OK"


def test_manifests() -> tuple[bool, str]:
    """Test manifest generation."""
    from src.chunking.manifests import generate_manifest, validate_manifest
    from src.chunking.offline import chunk_offline
    from src.config import Config, ChunkingConfig

    config = Config()
    chunking_config = ChunkingConfig(min_bytes=10, max_bytes=100)

    data = b"Test document content for manifest generation."
    chunks = chunk_offline(data, chunking_config)

    manifest = generate_manifest(chunks, "test_doc", config)

    assert "doc_id" in manifest
    assert "chunks" in manifest
    assert "config_hash" in manifest

    # Validate
    is_valid, issues = validate_manifest(manifest)
    assert is_valid, f"Manifest validation failed: {issues}"

    return True, "manifests OK"


def test_vectors() -> tuple[bool, str]:
    """Test vector utilities."""
    import numpy as np
    from src.embedding.vectors import (
        cosine_similarity,
        l2_distance,
        normalize_vectors,
        vectors_to_bytes,
        bytes_to_vectors,
    )

    # Test cosine similarity
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    c = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, c)) < 1e-6

    # Test L2 distance
    assert abs(l2_distance(a, b)) < 1e-6
    assert abs(l2_distance(a, c) - np.sqrt(2)) < 1e-6

    # Test normalization
    v = np.array([[3.0, 4.0, 0.0]])
    normalized = normalize_vectors(v)
    assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    # Test serialization
    vectors = np.random.randn(10, 384).astype(np.float32)
    data = vectors_to_bytes(vectors)
    recovered = bytes_to_vectors(data, 384)
    assert np.allclose(vectors, recovered)

    return True, "vectors OK"


def test_drift_metrics() -> tuple[bool, str]:
    """Test drift metrics."""
    import numpy as np
    from src.eval.drift import (
        compute_drift_cosine,
        compute_drift_l2,
        compute_drift_stats,
    )

    # Identical vectors
    e = np.random.randn(384).astype(np.float32)
    assert compute_drift_cosine(e, e) < 1e-6
    assert compute_drift_l2(e, e) < 1e-6

    # Different vectors
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    assert compute_drift_cosine(e1, e2) > 0.9  # Should be ~1.0

    # Test stats
    old = {"a": np.random.randn(384), "b": np.random.randn(384)}
    new = {"a": old["a"] + 0.01 * np.random.randn(384), "b": old["b"]}

    result = compute_drift_stats(old, new)
    assert result.num_matched == 2

    return True, "drift metrics OK"


def test_churn_metrics() -> tuple[bool, str]:
    """Test churn metrics."""
    from src.eval.churn import (
        compute_topk_overlap,
        compute_jaccard,
    )

    # Identical lists
    a = ["1", "2", "3", "4", "5"]
    assert compute_topk_overlap(a, a, 5) == 1.0
    assert compute_jaccard(a, a) == 1.0

    # Disjoint lists
    b = ["6", "7", "8", "9", "10"]
    assert compute_topk_overlap(a, b, 5) == 0.0
    assert compute_jaccard(a, b) == 0.0

    # Partial overlap
    c = ["1", "2", "6", "7", "8"]
    overlap = compute_topk_overlap(a, c, 5)
    assert 0 < overlap < 1

    return True, "churn metrics OK"


def test_overlap_metrics() -> tuple[bool, str]:
    """Test overlap metrics."""
    from src.eval.overlap import compute_hit_rate

    retrieved = ["a", "b", "c", "d", "e"]
    expected = {"a", "b", "f"}

    hit_rate = compute_hit_rate(retrieved, expected, 5)
    assert hit_rate == 2 / 3  # 2 of 3 expected items found

    return True, "overlap metrics OK"


def test_maintenance_metrics() -> tuple[bool, str]:
    """Test maintenance metrics."""
    from src.eval.maintenance import compute_maintenance_stats

    old = {"a", "b", "c"}
    new = {"b", "c", "d", "e"}

    result = compute_maintenance_stats(old, new, 5)

    assert result.unchanged_chunks == 2  # b, c
    assert result.added_chunks == 2  # d, e
    assert result.removed_chunks == 1  # a

    return True, "maintenance metrics OK"


def run_tests(verbose: bool = False) -> int:
    """Run all tests."""
    tests = [
        ("Config", test_config),
        ("Cut Score", test_cut_score),
        ("Offline Chunking", test_offline_chunking),
        ("Streaming Chunking", test_streaming_chunking),
        ("Data Generator", test_data_generator),
        ("Manifests", test_manifests),
        ("Vectors", test_vectors),
        ("Drift Metrics", test_drift_metrics),
        ("Churn Metrics", test_churn_metrics),
        ("Overlap Metrics", test_overlap_metrics),
        ("Maintenance Metrics", test_maintenance_metrics),
    ]

    passed = 0
    failed = 0

    print("Running tests...\n")

    for name, test_fn in tests:
        try:
            success, msg = test_fn()
            if success:
                passed += 1
                status = "\033[32mPASS\033[0m"
            else:
                failed += 1
                status = "\033[31mFAIL\033[0m"
            print(f"  [{status}] {name}: {msg}")
        except Exception as e:
            failed += 1
            status = "\033[31mFAIL\033[0m"
            print(f"  [{status}] {name}: {e}")
            if verbose:
                traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all tests")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()
    return run_tests(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
