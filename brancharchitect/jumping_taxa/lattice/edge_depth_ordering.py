"""
Edge Depth Ordering

This module provides depth computation and hierarchical ordering for lattice edges.
Orders edges based on subset relationships and tree depth to ensure optimal
processing: smaller, simpler conflicts are resolved before larger, complex ones.

Key principle: Process subsets before supersets
    Example: {A} → {B} → {A,B} → {A,B,C}

This incremental ordering is critical for the lattice algorithm's approach:
resolving simple conflicts first may simplify or eliminate larger conflicts.

Functions:
    - count_proper_subsets: Count subset relationships
    - compute_pivot_edge_depths: Calculate depth values respecting hierarchy
    - sort_pivot_edges_by_subset_hierarchy: Sort edges for optimal processing
"""

from __future__ import annotations
from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import is_full_overlap
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.debug import jt_logger


def count_proper_subsets(edge: Partition, pivot_edges: List[Partition]) -> int:
    """
    Count how many other partitions in the list are proper subsets of this edge.

    A partition with a higher count is a superset to more other partitions
    and should be processed later in the ordering. A proper subset means
    the other edge's indices are fully contained within this edge's indices,
    but they are not identical.

    Args:
        edge: The partition to check
        pivot_edges: List of all partitions to compare against

    Returns:
        Number of partitions that are proper subsets of the given edge

    Example:
        >>> edge = Partition({A, B, C})
        >>> pivot_edges = [Partition({A}), Partition({A,B}), Partition({B,C})]
        >>> count_proper_subsets(edge, pivot_edges)
        2  # {A,B} is a subset, {A} is a subset (but not {B,C})
    """
    superset_of_count = 0
    for other_edge in pivot_edges:
        if edge != other_edge:
            # Check if other_edge is a proper subset of edge
            if is_full_overlap(other_edge, edge):
                superset_of_count += 1
    return superset_of_count


def compute_pivot_edge_depths(
    pivot_edges: List[Partition], target: Node
) -> Dict[Partition, float]:
    """
    Compute depths for pivot edges with subset-based ordering.

    This function computes depths that respect subset relationships:
    - Subsets get smaller depth values (processed first)
    - Tree depth is used as a secondary ordering criterion
    - Missing edges get depth 0

    The ordering ensures {A} comes before {A,B} and {E} comes before {D,E}.

    Depth calculation formula:
        final_depth = (superset_of_count * 1000) + (partition_size * 10) + tree_depth

    This ensures:
        1. Subsets are processed before their supersets (superset_of_count factor)
        2. Smaller partitions are processed before larger ones (size factor)
        3. Tree depth provides fine-grained ordering within same level

    Args:
        pivot_edges: List of partitions to compute depths for
        target: Tree node to use for base depth calculation

    Returns:
        Dictionary mapping each partition to its computed depth value

    Example:
        For partitions {A}, {B}, {A,B}:
        - {A}: depth = 0*1000 + 1*10 + tree_depth = 10 + tree_depth
        - {B}: depth = 0*1000 + 1*10 + tree_depth = 10 + tree_depth
        - {A,B}: depth = 2*1000 + 2*10 + tree_depth = 2020 + tree_depth
        Result: {A} and {B} before {A,B}
    """
    pivot_edge_sort_map: Dict[Partition, float] = {}
    missing_edges: List[Partition] = []

    # First pass: collect basic tree depths
    basic_depths: Dict[Partition, float] = {}
    for edge in pivot_edges:
        node: Node | None = target.find_node_by_split(edge)
        if node and node.depth is not None:
            basic_depths[edge] = node.depth
        else:
            basic_depths[edge] = 0
            missing_edges.append(edge)

    # Second pass: compute subset-aware depths
    for edge in pivot_edges:
        tree_depth: float = basic_depths[edge]
        partition_size = len(edge)
        # Count how many other edges are proper subsets of this one.
        superset_of_count = count_proper_subsets(edge, pivot_edges)

        # Compute final depth with subset-based ordering.
        # Partitions that are supersets of others get a higher depth value,
        # ensuring they are processed later.
        superset_depth = superset_of_count * 1000  # Primary ordering factor
        size_depth = partition_size * 10  # Secondary ordering factor

        final_depth = superset_depth + size_depth + tree_depth
        pivot_edge_sort_map[edge] = final_depth

    return pivot_edge_sort_map


def sort_pivot_edges_by_subset_hierarchy(
    pivot_edges: List[PivotEdgeSubproblem], tree1: Node, tree2: Node
) -> List[PivotEdgeSubproblem]:
    """
    Sort pivot edges by subset hierarchy using tree depth for optimal processing order.

    Ensures that smaller, simpler conflicts are resolved before larger, more complex ones.
    This ordering is critical for the lattice algorithm's incremental approach:

    - Subsets processed before supersets (e.g., {A} → {A,B} → {A,B,C})
    - Tree depth provides fine-grained ordering within same subset level
    - Smaller partitions processed first, potentially simplifying larger conflicts

    The sorting uses average depth across both trees to balance the importance
    of each tree's structure in determining processing order.

    Args:
        pivot_edges: List of PivotEdgeSubproblem objects to sort
        tree1: First phylogenetic tree for depth calculation
        tree2: Second phylogenetic tree for depth calculation

    Returns:
        Sorted list of pivot edges in ascending order (subsets first)

    Example:
        Given edges for partitions {A}, {A,B}, {B}:
        Returns them ordered as: [{A}, {B}, {A,B}]
    """
    if not pivot_edges:
        return pivot_edges

    # Extract partitions from pivot edges
    partitions = [edge.pivot_split for edge in pivot_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_pivot_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_pivot_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths: Dict[Partition, float] = {}
    for partition in partitions:
        avg_depths[partition] = (depths1[partition] + depths2[partition]) / 2

    # Sort pivot edges by their average depths (ascending = subsets first)
    sorted_edges: List[PivotEdgeSubproblem] = sorted(
        pivot_edges,
        key=lambda edge: avg_depths[edge.pivot_split],
    )

    jt_logger.debug(
        f"Sorted {len(pivot_edges)} pivot edges by subset hierarchy and depth"
    )
    for i, edge in enumerate(sorted_edges):
        depth = avg_depths[edge.pivot_split]
        jt_logger.debug(f"  {i + 1}. {edge.pivot_split} (avg_depth={depth})")

    return sorted_edges
