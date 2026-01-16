"""
Maintenance cost metrics for curv-embedding.

Tracks the operational cost of maintaining embeddings across content updates,
including re-embedding requirements, index rebuilds, and tombstone accumulation.

Metrics:
- reembed_fraction: Fraction of chunks requiring re-embedding
- added_chunks: New chunks not in previous version
- removed_chunks: Chunks deleted from previous version
- unchanged_chunks: Chunks preserved without modification
- tombstone_rate: Fraction of index entries that are tombstones
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaintenanceResult:
    """
    Maintenance cost statistics for an embedding update cycle.

    Captures the work required to update embeddings when content changes,
    useful for evaluating chunking strategy efficiency.

    Attributes:
        reembed_fraction: Fraction of total chunks that needed re-embedding.
            Lower is better (more stability).
        added_chunks: Number of new chunks introduced.
        removed_chunks: Number of chunks removed (became tombstones).
        unchanged_chunks: Number of chunks preserved from previous version.
        total_chunks_old: Total chunks in old version.
        total_chunks_new: Total chunks in new version.
        tombstone_count: Number of tombstone entries in index.
        tombstone_rate: Fraction of index entries that are tombstones.
        index_rebuild_required: Whether a full index rebuild was needed.
        index_rebuild_events: Count of rebuild events in this update cycle.
    """

    reembed_fraction: float
    added_chunks: int
    removed_chunks: int
    unchanged_chunks: int
    total_chunks_old: int
    total_chunks_new: int
    tombstone_count: int
    tombstone_rate: float
    index_rebuild_required: bool
    index_rebuild_events: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "reembed_fraction": self.reembed_fraction,
            "added_chunks": self.added_chunks,
            "removed_chunks": self.removed_chunks,
            "unchanged_chunks": self.unchanged_chunks,
            "total_chunks_old": self.total_chunks_old,
            "total_chunks_new": self.total_chunks_new,
            "tombstone_count": self.tombstone_count,
            "tombstone_rate": self.tombstone_rate,
            "index_rebuild_required": self.index_rebuild_required,
            "index_rebuild_events": self.index_rebuild_events,
        }


def compute_maintenance_stats(
    old_chunks: set[str],
    new_chunks: set[str],
    total_chunks: int | None = None,
    existing_tombstones: int = 0,
    tombstone_threshold: float = 0.2,
) -> MaintenanceResult:
    """
    Compute maintenance statistics for a content update.

    Compares chunk sets (by content SHA256) between old and new versions
    to determine the maintenance cost of the update.

    Args:
        old_chunks: Set of chunk content hashes in the old version.
        new_chunks: Set of chunk content hashes in the new version.
        total_chunks: Optional total chunk count for denominator.
            If None, uses len(new_chunks).
        existing_tombstones: Number of existing tombstones before this update.
        tombstone_threshold: Fraction of tombstones that triggers rebuild.
            Default 0.2 (20%).

    Returns:
        MaintenanceResult with computed statistics.
    """
    # Set operations
    unchanged = old_chunks & new_chunks
    added = new_chunks - old_chunks
    removed = old_chunks - new_chunks

    unchanged_count = len(unchanged)
    added_count = len(added)
    removed_count = len(removed)

    total_old = len(old_chunks)
    total_new = len(new_chunks)

    # Determine denominator for reembed fraction
    if total_chunks is None:
        total_chunks = total_new

    # Reembed fraction: chunks that changed / total chunks
    # Changed chunks = added chunks (new content needs embedding)
    # Note: removed chunks don't need re-embedding, they just become tombstones
    if total_chunks > 0:
        reembed_fraction = added_count / total_chunks
    else:
        reembed_fraction = 0.0

    # Tombstone tracking
    # New tombstones from this update = removed chunks
    tombstone_count = existing_tombstones + removed_count

    # Tombstone rate: tombstones / (active + tombstones)
    total_index_entries = total_new + tombstone_count
    if total_index_entries > 0:
        tombstone_rate = tombstone_count / total_index_entries
    else:
        tombstone_rate = 0.0

    # Determine if rebuild is needed
    index_rebuild_required = tombstone_rate >= tombstone_threshold
    index_rebuild_events = 1 if index_rebuild_required else 0

    return MaintenanceResult(
        reembed_fraction=reembed_fraction,
        added_chunks=added_count,
        removed_chunks=removed_count,
        unchanged_chunks=unchanged_count,
        total_chunks_old=total_old,
        total_chunks_new=total_new,
        tombstone_count=tombstone_count,
        tombstone_rate=tombstone_rate,
        index_rebuild_required=index_rebuild_required,
        index_rebuild_events=index_rebuild_events,
    )


def compute_cumulative_maintenance(
    chunk_versions: list[set[str]],
    tombstone_threshold: float = 0.2,
) -> list[MaintenanceResult]:
    """
    Compute maintenance statistics across multiple version updates.

    Tracks cumulative maintenance cost as content evolves through
    multiple revisions.

    Args:
        chunk_versions: List of chunk sets (by content SHA256) for each version.
            Must have at least 2 versions.
        tombstone_threshold: Fraction of tombstones that triggers rebuild.

    Returns:
        List of MaintenanceResult, one for each version transition.

    Raises:
        ValueError: If fewer than 2 versions provided.
    """
    if len(chunk_versions) < 2:
        raise ValueError("Need at least 2 versions to compute maintenance stats")

    results: list[MaintenanceResult] = []
    cumulative_tombstones = 0

    for i in range(1, len(chunk_versions)):
        old_chunks = chunk_versions[i - 1]
        new_chunks = chunk_versions[i]

        result = compute_maintenance_stats(
            old_chunks=old_chunks,
            new_chunks=new_chunks,
            existing_tombstones=cumulative_tombstones,
            tombstone_threshold=tombstone_threshold,
        )

        results.append(result)

        # Update cumulative tombstones
        if result.index_rebuild_required:
            # Rebuild clears tombstones
            cumulative_tombstones = 0
        else:
            cumulative_tombstones = result.tombstone_count

    return results


@dataclass
class MaintenanceSummary:
    """
    Summary statistics across multiple version updates.

    Attributes:
        mean_reembed_fraction: Mean reembed fraction across updates.
        max_reembed_fraction: Maximum reembed fraction (worst case).
        total_rebuild_events: Total number of index rebuilds required.
        mean_tombstone_rate: Mean tombstone rate across updates.
        total_added: Total chunks added across all updates.
        total_removed: Total chunks removed across all updates.
        num_updates: Number of version transitions analyzed.
    """

    mean_reembed_fraction: float
    max_reembed_fraction: float
    total_rebuild_events: int
    mean_tombstone_rate: float
    total_added: int
    total_removed: int
    num_updates: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_reembed_fraction": self.mean_reembed_fraction,
            "max_reembed_fraction": self.max_reembed_fraction,
            "total_rebuild_events": self.total_rebuild_events,
            "mean_tombstone_rate": self.mean_tombstone_rate,
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "num_updates": self.num_updates,
        }


def summarize_maintenance(results: list[MaintenanceResult]) -> MaintenanceSummary:
    """
    Compute summary statistics from a list of maintenance results.

    Args:
        results: List of MaintenanceResult from compute_cumulative_maintenance.

    Returns:
        MaintenanceSummary with aggregated statistics.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot summarize empty results list")

    reembed_fractions = [r.reembed_fraction for r in results]
    tombstone_rates = [r.tombstone_rate for r in results]

    return MaintenanceSummary(
        mean_reembed_fraction=sum(reembed_fractions) / len(reembed_fractions),
        max_reembed_fraction=max(reembed_fractions),
        total_rebuild_events=sum(r.index_rebuild_events for r in results),
        mean_tombstone_rate=sum(tombstone_rates) / len(tombstone_rates),
        total_added=sum(r.added_chunks for r in results),
        total_removed=sum(r.removed_chunks for r in results),
        num_updates=len(results),
    )
