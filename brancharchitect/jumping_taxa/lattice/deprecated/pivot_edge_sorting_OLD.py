"""
Pivot Edge Sorting

Sorts pivot edges by subset hierarchy to ensure optimal processing order.
Smaller partitions (subsets) are processed before larger ones (supersets),
enabling incremental solution building from simple to complex conflicts.

Key concept: Processing {A} before {A,B} allows simpler conflicts to be
resolved first, potentially simplifying or eliminating larger conflicts.
"""

from typing import List, Dict

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)
from brancharchitect.jumping_taxa.debug import jt_logger


def sort_lattice_edges_by_subset_hierarchy(
    lattice_edges: List[PivotEdgeSubproblem], tree1: Node, tree2: Node
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
        lattice_edges: List of PivotEdgeSubproblem objects to sort
        tree1: First phylogenetic tree for depth calculation
        tree2: Second phylogenetic tree for depth calculation

    Returns:
        Sorted list of pivot edges in ascending order (subsets first)

    Example:
        Given edges for partitions {A}, {A,B}, {B}:
        Returns them ordered as: [{A}, {B}, {A,B}]
    """
    if not lattice_edges:
        return lattice_edges

    # Extract partitions from lattice edges
    partitions = [edge.pivot_split for edge in lattice_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths: Dict[Partition, float] = {}
    for partition in partitions:
        avg_depths[partition] = (depths1[partition] + depths2[partition]) / 2

    # Sort lattice edges by their average depths (ascending = subsets first)
    sorted_edges: List[PivotEdgeSubproblem] = sorted(
        lattice_edges,
        key=lambda edge: avg_depths[edge.pivot_split],
    )

    jt_logger.debug(
        f"Sorted {len(lattice_edges)} lattice edges by subset hierarchy and depth"
    )
    for i, edge in enumerate(sorted_edges):
        depth = avg_depths[edge.pivot_split]
        jt_logger.debug(f"  {i + 1}. {edge.pivot_split} (avg_depth={depth})")

    return sorted_edges
