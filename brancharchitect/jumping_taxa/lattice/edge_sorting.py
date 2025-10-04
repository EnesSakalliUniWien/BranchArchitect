"""Lattice edge sorting utilities."""

from typing import List, Dict

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)
from brancharchitect.jumping_taxa.debug import jt_logger


def sort_lattice_edges_by_subset_hierarchy(
    lattice_edges: List[LatticeEdge], tree1: Node, tree2: Node
) -> List[LatticeEdge]:
    """
    Sort lattice edges by subset hierarchy and tree depth, ensuring smaller sets are processed first.

    Uses the same ordering logic as tree interpolation depth calculation:
    - Subsets are processed before their supersets (e.g., {A} before {A,B})
    - Tree depth provides fine-grained ordering within same subset level
    - Smaller partitions are processed before larger ones

    Args:
        lattice_edges: List of LatticeEdge objects to sort
        tree1: First tree for depth calculation
        tree2: Second tree for depth calculation

    Returns:
        Sorted list of LatticeEdge objects in ascending subset order
    """
    if not lattice_edges:
        return lattice_edges

    # Extract partitions from lattice edges
    partitions = [edge.split for edge in lattice_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths: Dict[Partition, float] = {}
    for partition in partitions:
        avg_depths[partition] = (depths1[partition] + depths2[partition]) / 2

    # Sort lattice edges by their average depths (ascending = subsets first)
    sorted_edges: List[LatticeEdge] = sorted(
        lattice_edges,
        key=lambda edge: avg_depths[edge.split],
    )

    jt_logger.debug(
        f"Sorted {len(lattice_edges)} lattice edges by subset hierarchy and depth"
    )
    for i, edge in enumerate(sorted_edges):
        depth = avg_depths[edge.split]
        jt_logger.debug(f"  {i + 1}. {edge.split} (avg_depth={depth})")

    return sorted_edges
