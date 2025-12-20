"""
Edge Sorting Utilities

Simple utility functions for sorting lattice edges by depth and hierarchy.
Replaces the overly complex LatticeEdgeData class with straightforward functions.
"""

from __future__ import annotations
from typing import List, Dict
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    compute_pivot_edge_depths,
)


def sort_edges_by_depth(
    edges: List[Partition], tree: Node, ascending: bool = True
) -> List[Partition]:
    """
    Sort edges by their hierarchical depth in the tree with deterministic tie-breaking.

    Sorting criteria (in priority order):
    1. Topological Rank (primary): Rank from topological sort where subsets < supersets.
       (Ensures strict subset-before-superset ordering).
    2. Partition size (secondary): Smaller partitions first.
    3. Sorted indices (deterministic tiebreaker): Lexicographical order of indices.

    Args:
        edges: List of partitions to sort
        tree: Tree to compute depths from
        ascending: If True, sort by increasing rank (Subsets -> Supersets / Leaves -> Root).
                  If False, sort by decreasing rank (Supersets -> Subsets / Root -> Leaves).

    Returns:
        Sorted list of edges

    Example:
        >>> edges = [partition_AB, partition_A, partition_B, partition_ABC]
        >>> sorted_edges = sort_edges_by_depth(edges, tree, ascending=True)
        >>> # Result: [partition_A, partition_B, partition_AB, partition_ABC]
        >>> # (leaves first, then their parents)
    """
    # Compute depths for all edges
    depths: Dict[Partition, float] = compute_pivot_edge_depths(edges, tree)

    # Sort with multi-level key for deterministic ordering:
    # (depth, partition size, sorted indices tuple)
    return sorted(
        edges,
        key=lambda edge: (
            depths.get(edge, 0),
            len(edge),
            tuple(sorted(int(i) for i in edge)),
        ),
        reverse=not ascending,
    )


def compute_edge_depths(edges: List[Partition], tree: Node) -> Dict[Partition, float]:
    """
    Compute depth values for a list of edges.

    This is a simple wrapper around compute_pivot_edge_depths for convenience.

    Args:
        edges: List of partitions to compute depths for
        tree: Tree to use for depth calculation

    Returns:
        Dictionary mapping each partition to its depth value
    """
    return compute_pivot_edge_depths(edges, tree)
