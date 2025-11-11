from __future__ import annotations

"""
Frontier utilities for subtree-shared split analysis.

This module provides helpers to access, for a given child subtree and a set of
"across-trees" shared splits under a pivot, the maximal shared elements
("frontier"), the shared downset under each frontier element, the minimal
elements of that downset, and its bottom when unique.

It also provides the nested mapping structure requested:

    frontier_map: dict[
        Partition (frontier_max),
        dict[Partition (bottom), FrontierToBottom]
    ]

All comparisons are index/bitmask-based and encoding-agnostic, consistent with
the core Partition/PartitionSet semantics used elsewhere.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from brancharchitect.elements.partition_set import PartitionSet, Partition
from brancharchitect.elements.frozen_partition_set import FrozenPartitionSet

if TYPE_CHECKING:
    from brancharchitect.tree import Node


def bottom_or_none(ps: PartitionSet[Partition]) -> Optional[Partition]:
    """Return the unique minimal element of ``ps`` if it exists, else None."""
    mins = ps.minimal_elements()
    if len(mins) == 1:
        for p in mins:
            return p
    return None


def downset_under(frontier_max: Partition, ps: PartitionSet[Partition]) -> PartitionSet[Partition]:
    """Return the downset of ``ps`` under the given ``frontier_max`` (all s ⊆ frontier_max)."""
    fm = frontier_max.bitmask
    elems = {p for p in ps if (p.bitmask & ~fm) == 0}
    return type(ps)(splits=elems, encoding=ps.encoding, name=f"{ps.name}_downset", order=ps.order)


def map_frontier_to_minimals_by_shared_set(
    child: "Node",
    across_trees_splits_under_pivot_with_leaves: PartitionSet[Partition],
) -> Dict[FrozenPartitionSet[Partition], Dict[str, Any]]:
    """
    Build a mapping from each frontier element's shared downset (as the key)
    to its minimal elements and bottom, for a given child subtree.
    Key is a FrozenPartitionSet(shared_under_frontier).
    """
    child_splits_across_trees: PartitionSet[Partition] = (
        child.to_splits(with_leaves=True) & across_trees_splits_under_pivot_with_leaves
    )
    child_frontier: PartitionSet[Partition] = child_splits_across_trees.maximal_elements()

    result: Dict[FrozenPartitionSet[Partition], Dict[str, Any]] = {}

    for frontier_max in child_frontier:
        shared_under = downset_under(frontier_max, child_splits_across_trees)
        minimals = shared_under.minimal_elements()
        bottom = bottom_or_none(shared_under)

        key = FrozenPartitionSet(
            splits=set(shared_under),
            encoding=shared_under.encoding,
            name="shared_under_frontier",
            order=shared_under.order,
        )
        result[key] = {
            "frontier_max": frontier_max,
            "shared_under_frontier": shared_under,
            "minimals": minimals,
            "bottom": bottom,
        }

    return result


@dataclass(frozen=True)
class ChildFrontierInfo:
    """Details for a single frontier (maximal shared) element."""

    shared_under_frontier: PartitionSet[Partition]
    minimals: PartitionSet[Partition]
    bottom: Optional[Partition]


@dataclass(frozen=True)
class FrontierToBottom:
    """Details for a bottom under a specific frontier.

    - shared_under_bottom: all shared splits s ⊆ bottom (typically just {bottom})
    - shared_at_frontier: all shared splits s ⊆ frontier (the frontier's downset)
    """

    shared_under_bottom: PartitionSet[Partition]
    shared_at_frontier: PartitionSet[Partition]


def build_nested_frontier_map(
    child: "Node",
    across_trees_splits_under_pivot_with_leaves: PartitionSet[Partition],
) -> Dict[Partition, Dict[Partition, FrontierToBottom]]:
    """
    Build a nested mapping per child subtree:

        frontier_map: dict[
            Partition (frontier_max),
            dict[Partition (bottom), FrontierToBottom]
        ]

    Where:
      - frontier_max are the maximal shared (frontier) elements in the child's subtree
      - bottom are minimal elements (bottoms) within the frontier's shared downset
      - FrontierToBottom carries both the shared-under-bottom downset and the shared-at-frontier downset
    """
    # Shared splits in this child's subtree (across trees, under pivot)
    child_splits_across_trees: PartitionSet[Partition] = (
        child.to_splits(with_leaves=True) & across_trees_splits_under_pivot_with_leaves
    )
    # Frontier (maximal shared elements)
    child_frontiers: PartitionSet[Partition] = child_splits_across_trees.maximal_elements()

    nested: Dict[Partition, Dict[Partition, FrontierToBottom]] = {}

    for frontier_max in child_frontiers:
        shared_at_frontier = downset_under(frontier_max, child_splits_across_trees)
        bottoms = shared_at_frontier.minimal_elements()
        inner: Dict[Partition, FrontierToBottom] = {}
        for bottom_min in bottoms:
            shared_under_bottom = downset_under(bottom_min, child_splits_across_trees)
            inner[bottom_min] = FrontierToBottom(
                shared_under_bottom=shared_under_bottom,
                shared_at_frontier=shared_at_frontier,
            )
        nested[frontier_max] = inner

    return nested


# ---------------------------------------------------------------------------
# Top → Bottom mapping with a class bundle (requested API)
#   top_map: dict[Partition (frontier_max), TopToBottom]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopToBottom:
    """Bundle of shared splits at a frontier and under each bottom.

    - shared_top_splits: downset under the frontier (all shared s ⊆ frontier)
    - bottoms: mapping bottom → shared_under_bottom (downset under bottom)
    """

    shared_top_splits: PartitionSet[Partition]
    bottoms: Dict[Partition, PartitionSet[Partition]]


def build_top_to_bottom(
    child: "Node",
    across_trees_splits_under_pivot_with_leaves: PartitionSet[Partition],
) -> Dict[Partition, TopToBottom]:
    """
    Build a mapping per child subtree:

        dict[Partition (frontier_max), TopToBottom]

    This keeps the same elements you're already using:
      - child_frontiers = maximal_elements(child_splits_across_trees)
      - bottoms = minimal_elements(shared_at_frontier)
      - shared_under_bottom computed from the child's shared set by subset mask
    """
    # Shared splits in this child's subtree (across trees, under pivot)
    child_splits_across_trees: PartitionSet[Partition] = (
        child.to_splits(with_leaves=True) & across_trees_splits_under_pivot_with_leaves
    )
    # Frontier (maximal shared elements)
    child_frontiers: PartitionSet[Partition] = child_splits_across_trees.maximal_elements()

    result: Dict[Partition, TopToBottom] = {}

    for frontier_max in child_frontiers:
        # Downset under frontier
        fmask = frontier_max.bitmask
        shared_at_frontier_elems = {s for s in child_splits_across_trees if (s.bitmask & ~fmask) == 0}
        shared_at_frontier = type(child_splits_across_trees)(
            splits=shared_at_frontier_elems,
            encoding=child_splits_across_trees.encoding,
            name="shared_at_frontier",
            order=child_splits_across_trees.order,
        )
        # Bottoms (minimal elements) under frontier
        bottoms = shared_at_frontier.minimal_elements()

        bottoms_map: Dict[Partition, PartitionSet[Partition]] = {}
        for bottom in bottoms:
            bmask = bottom.bitmask
            shared_under_bottom_elems = {s for s in child_splits_across_trees if (s.bitmask & ~bmask) == 0}
            shared_under_bottom = type(child_splits_across_trees)(
                splits=shared_under_bottom_elems,
                encoding=child_splits_across_trees.encoding,
                name="shared_under_bottom",
                order=child_splits_across_trees.order,
            )
            bottoms_map[bottom] = shared_under_bottom

        result[frontier_max] = TopToBottom(
            shared_top_splits=shared_at_frontier,
            bottoms=bottoms_map,
        )

    return result
