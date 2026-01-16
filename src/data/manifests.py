"""
Data manifest utilities for curv-embedding evaluation.

Provides metadata generation and query family structures for synthetic data.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import Config
    from src.data.generator import SyntheticDocument


@dataclass
class DataManifest:
    """Metadata for a synthetic corpus.

    Attributes:
        seed: Random seed used for generation.
        num_documents: Total number of documents.
        domains: List of domains in the corpus.
        total_bytes: Total size of all documents.
        doc_ids: List of document IDs.
        boundary_counts: Dict mapping doc_id to boundary count.
        anchor_counts: Dict mapping doc_id to anchor count.
        config_hash: Hash of the configuration used.
    """

    seed: int
    num_documents: int
    domains: list[str]
    total_bytes: int
    doc_ids: list[str]
    boundary_counts: dict[str, int] = field(default_factory=dict)
    anchor_counts: dict[str, int] = field(default_factory=dict)
    config_hash: str = ""


@dataclass
class QueryFamily:
    """A family of reformulated queries targeting the same anchors.

    Attributes:
        family_id: Unique identifier for this family.
        queries: List of query strings (reformulations).
        target_anchor_ids: List of anchor IDs these queries should retrieve.
        source_domain: Domain of the source document.
        source_doc_id: ID of the source document.
    """

    family_id: str
    queries: list[str]
    target_anchor_ids: list[str]
    source_domain: str = ""
    source_doc_id: str = ""


def generate_data_manifest(
    documents: list[SyntheticDocument],
    seed: int,
    config: Config | None = None,
) -> dict[str, Any]:
    """Generate a data manifest from a list of documents.

    Args:
        documents: List of SyntheticDocument instances.
        seed: Random seed used for generation.
        config: Optional configuration object.

    Returns:
        Dictionary containing manifest data suitable for JSON serialization.
    """
    domains = sorted(set(doc.domain for doc in documents))
    total_bytes = sum(len(doc.content) for doc in documents)
    doc_ids = [doc.doc_id for doc in documents]

    boundary_counts = {doc.doc_id: len(doc.boundary_offsets) for doc in documents}
    anchor_counts = {doc.doc_id: len(doc.planted_anchors) for doc in documents}

    config_hash = config.config_hash() if config else ""

    manifest = DataManifest(
        seed=seed,
        num_documents=len(documents),
        domains=domains,
        total_bytes=total_bytes,
        doc_ids=doc_ids,
        boundary_counts=boundary_counts,
        anchor_counts=anchor_counts,
        config_hash=config_hash,
    )

    return _manifest_to_dict(manifest)


def _manifest_to_dict(manifest: DataManifest) -> dict[str, Any]:
    """Convert DataManifest to dictionary for JSON serialization."""
    return {
        "seed": manifest.seed,
        "num_documents": manifest.num_documents,
        "domains": manifest.domains,
        "total_bytes": manifest.total_bytes,
        "doc_ids": manifest.doc_ids,
        "boundary_counts": manifest.boundary_counts,
        "anchor_counts": manifest.anchor_counts,
        "config_hash": manifest.config_hash,
        "summary": {
            "total_boundaries": sum(manifest.boundary_counts.values()),
            "total_anchors": sum(manifest.anchor_counts.values()),
            "avg_doc_size": manifest.total_bytes // max(manifest.num_documents, 1),
            "domain_distribution": _compute_domain_distribution(manifest),
        },
    }


def _compute_domain_distribution(manifest: DataManifest) -> dict[str, int]:
    """Compute count of documents per domain."""
    distribution: dict[str, int] = {d: 0 for d in manifest.domains}
    for doc_id in manifest.doc_ids:
        # Extract domain from doc_id (format: domain_hexseed)
        domain = doc_id.split("_")[0]
        if domain in distribution:
            distribution[domain] += 1
    return distribution


# Query generation templates
_TEXT_QUERY_TEMPLATES = [
    "What does the text say about {topic}?",
    "Find information regarding {topic}",
    "Search for content mentioning {topic}",
    "Retrieve passages about {topic}",
    "Look up {topic} in the documents",
    "What is written about {topic}?",
    "Find text discussing {topic}",
    "Show me content related to {topic}",
]

_CODE_QUERY_TEMPLATES = [
    "Find the function that handles {topic}",
    "Show the code for {topic}",
    "Where is {topic} implemented?",
    "Find the implementation of {topic}",
    "Show me the {topic} function",
    "Look up the code for {topic}",
    "Find where {topic} is defined",
    "Show the {topic} handler",
]

_JSON_QUERY_TEMPLATES = [
    "Find the record with {topic}",
    "Show the entry containing {topic}",
    "Look up {topic} in the data",
    "Find objects with {topic}",
    "Retrieve the {topic} entry",
    "Show records matching {topic}",
    "Find data about {topic}",
    "Look up entries for {topic}",
]

_LOG_QUERY_TEMPLATES = [
    "Find log entries about {topic}",
    "Show logs mentioning {topic}",
    "Look up {topic} in the logs",
    "Find errors related to {topic}",
    "Show log messages for {topic}",
    "Find the {topic} log entries",
    "Look up logs containing {topic}",
    "Show events about {topic}",
]


def _extract_topic_from_content(
    content: bytes,
    anchor_start: int,
    anchor_end: int,
    rng: random.Random,
) -> str:
    """Extract a topic keyword from anchor content."""
    try:
        anchor_text = content[anchor_start:anchor_end].decode("utf-8", errors="ignore")
        # Extract words and pick one as topic
        words = [w.strip(".,;:!?\"'()[]{}") for w in anchor_text.split()]
        words = [w for w in words if len(w) >= 3 and w.isalnum()]
        if words:
            return rng.choice(words)
    except Exception:
        pass
    return f"topic_{rng.randint(1000, 9999)}"


def generate_query_families(
    documents: list[SyntheticDocument],
    num_families: int,
    reformulations_per_family: int,
    seed: int,
) -> list[QueryFamily]:
    """Generate query families for reformulation testing.

    Each family contains multiple reformulated queries that should all
    retrieve the same anchor(s).

    Args:
        documents: List of SyntheticDocument instances with planted anchors.
        num_families: Number of query families to generate.
        reformulations_per_family: Number of query reformulations per family.
        seed: Random seed for reproducibility.

    Returns:
        List of QueryFamily instances.
    """
    rng = random.Random(seed)

    # Collect all documents with anchors
    docs_with_anchors = [doc for doc in documents if doc.planted_anchors]
    if not docs_with_anchors:
        return []

    families = []
    templates_by_domain = {
        "text": _TEXT_QUERY_TEMPLATES,
        "code": _CODE_QUERY_TEMPLATES,
        "json": _JSON_QUERY_TEMPLATES,
        "logs": _LOG_QUERY_TEMPLATES,
    }

    for i in range(num_families):
        # Pick a random document with anchors
        doc = rng.choice(docs_with_anchors)

        # Pick a random anchor from that document
        anchor = rng.choice(doc.planted_anchors)
        anchor_start, anchor_end, anchor_id = anchor

        # Extract topic from anchor region
        topic = _extract_topic_from_content(doc.content, anchor_start, anchor_end, rng)

        # Get templates for this domain
        templates = templates_by_domain.get(doc.domain, _TEXT_QUERY_TEMPLATES)

        # Generate reformulations
        queries = []
        used_templates = rng.sample(
            templates,
            min(reformulations_per_family, len(templates)),
        )
        for template in used_templates:
            query = template.format(topic=topic)
            queries.append(query)

        # Add some simple variations
        while len(queries) < reformulations_per_family:
            base_query = rng.choice(queries[:len(used_templates)])
            # Simple variations: add/remove words, change phrasing
            variations = [
                base_query.lower(),
                base_query.upper(),
                f"Please {base_query.lower()}",
                f"{base_query.rstrip('?.')}?",
                f"I need to {base_query.lower().replace('find', 'locate')}",
            ]
            new_query = rng.choice(variations)
            if new_query not in queries:
                queries.append(new_query)

        family = QueryFamily(
            family_id=f"family_{i:04d}",
            queries=queries[:reformulations_per_family],
            target_anchor_ids=[anchor_id],
            source_domain=doc.domain,
            source_doc_id=doc.doc_id,
        )
        families.append(family)

    return families


def query_families_to_dict(families: list[QueryFamily]) -> list[dict[str, Any]]:
    """Convert query families to dictionary for JSON serialization."""
    return [
        {
            "family_id": f.family_id,
            "queries": f.queries,
            "target_anchor_ids": f.target_anchor_ids,
            "source_domain": f.source_domain,
            "source_doc_id": f.source_doc_id,
        }
        for f in families
    ]


def generate_anchors_manifest(documents: list[SyntheticDocument]) -> dict[str, Any]:
    """Generate anchor metadata manifest.

    Args:
        documents: List of SyntheticDocument instances.

    Returns:
        Dictionary containing all anchor metadata.
    """
    anchors = []
    for doc in documents:
        for start, end, anchor_id in doc.planted_anchors:
            # Extract anchor content preview
            try:
                preview = doc.content[start:min(end, start + 100)].decode("utf-8", errors="ignore")
            except Exception:
                preview = ""

            anchors.append({
                "anchor_id": anchor_id,
                "doc_id": doc.doc_id,
                "domain": doc.domain,
                "start_offset": start,
                "end_offset": end,
                "length": end - start,
                "preview": preview[:100] if len(preview) > 100 else preview,
            })

    return {
        "total_anchors": len(anchors),
        "anchors_by_domain": _count_by_domain(anchors),
        "anchors": anchors,
    }


def _count_by_domain(anchors: list[dict[str, Any]]) -> dict[str, int]:
    """Count anchors by domain."""
    counts: dict[str, int] = {}
    for anchor in anchors:
        domain = anchor.get("domain", "unknown")
        counts[domain] = counts.get(domain, 0) + 1
    return counts
