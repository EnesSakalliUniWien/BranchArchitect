"""
Sibling Grouping for Interpolation Frame Building.

Pre-computes which movers should be animated together based on shared
moving parents. Uses simple set membership - no complex data structures needed.

Phase-specific grouping:
- Collapse phase: group by source parent (if parent is collapsing)
- Expand phase: group by destination parent (if parent is expanding)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


def compute_sibling_groups(
    all_mover_partitions: List[Partition],
    collapse_splits: Set[Partition],
    expand_splits: Set[Partition],
    source_parent_map: Optional[Dict[Partition, Partition]],
    dest_parent_map: Optional[Dict[Partition, Partition]],
) -> Tuple[Dict[Partition, List[Partition]], Dict[Partition, List[Partition]]]:
    """
    Pre-compute which movers should be animated together, per phase.

    Collapse phase: siblings grouped by shared SOURCE parent (if collapsing).
    Expand phase: siblings grouped by shared DEST parent (if expanding).

    Args:
        all_mover_partitions: All movers for this pivot edge.
        collapse_splits: All splits that are collapsing (source-unique).
        expand_splits: All splits that are expanding (dest-unique).
        source_parent_map: Maps each mover -> its parent in source tree.
        dest_parent_map: Maps each mover -> its parent in destination tree.

    Returns:
        Tuple of (collapse_groups, expand_groups):
        - collapse_groups: Dict mapping mover -> siblings for collapse phase
        - expand_groups: Dict mapping mover -> siblings for expand phase
    """
    if not all_mover_partitions:
        return {}, {}

    collapse_groups = _build_phase_groups(
        all_mover_partitions, source_parent_map, collapse_splits
    )
    expand_groups = _build_phase_groups(
        all_mover_partitions, dest_parent_map, expand_splits
    )

    return collapse_groups, expand_groups


def _build_phase_groups(
    all_mover_partitions: List[Partition],
    parent_map: Optional[Dict[Partition, Partition]],
    moving_splits: Set[Partition],
) -> Dict[Partition, List[Partition]]:
    """
    Build sibling groups for a single phase.

    Args:
        all_mover_partitions: All movers for this pivot edge.
        parent_map: Maps each mover -> its parent in the relevant tree.
        moving_splits: Splits that are moving in this phase.

    Returns:
        Dict mapping each mover -> its sibling group (sorted list).
    """
    if not parent_map:
        # No parent info: everyone is a singleton
        return {m: [m] for m in all_mover_partitions}

    # Group movers by their moving parent
    parent_to_movers: Dict[Partition, List[Partition]] = {}

    for mover in all_mover_partitions:
        parent = parent_map.get(mover)
        if parent and parent in moving_splits:
            parent_to_movers.setdefault(parent, []).append(mover)

    # Build result: each mover maps to its group
    result: Dict[Partition, List[Partition]] = {}

    for _parent, siblings in parent_to_movers.items():
        # Sort for deterministic behavior (size desc, then bitmask)
        sorted_siblings = sorted(siblings, key=lambda p: (-p.size, p.bitmask))
        for mover in sorted_siblings:
            result[mover] = sorted_siblings

    # Movers without moving parents get themselves as singleton groups
    for mover in all_mover_partitions:
        if mover not in result:
            result[mover] = [mover]

    return result


def get_collapse_splits(
    collapse_paths: Dict[Partition, PartitionSet[Partition]],
) -> Set[Partition]:
    """
    Build the set of ALL partitions that are collapsing.
    """
    all_collapse: Set[Partition] = set()
    for paths in collapse_paths.values():
        all_collapse.update(paths)
    return all_collapse


def get_expand_splits(
    expand_paths: Dict[Partition, PartitionSet[Partition]],
) -> Set[Partition]:
    """
    Build the set of ALL partitions that are expanding.
    """
    all_expand: Set[Partition] = set()
    for paths in expand_paths.values():
        all_expand.update(paths)
    return all_expand


def get_group_for_mover(
    mover: Partition,
    sibling_groups: Dict[Partition, List[Partition]],
) -> List[Partition]:
    """
    Retrieve the sibling group for a mover.

    Falls back to [mover] if not found in pre-computed groups.
    """
    return sibling_groups.get(mover, [mover])

