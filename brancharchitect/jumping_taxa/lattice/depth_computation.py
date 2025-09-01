"""
Depth computation utilities for lattice edges.

This module contains functions for computing depths of lattice edges
with subset-based ordering for phylogenetic tree interpolation.
"""

from __future__ import annotations
from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import is_full_overlap
from brancharchitect.tree import Node


def _count_proper_subsets(edge: Partition, all_edges: List[Partition]) -> int:
    """
    Count how many other partitions in the list are proper subsets of this edge.

    A partition with a higher count is a superset to more other partitions
    and should be processed later in the ordering. A proper subset means
    the other edge's indices are fully contained within this edge's indices,
    but they are not identical.
    """
    superset_of_count = 0
    for other_edge in all_edges:
        if edge != other_edge:
            # Check if other_edge is a proper subset of edge
            if is_full_overlap(other_edge, edge):
                superset_of_count += 1
    return superset_of_count


def compute_lattice_edge_depths(
    lattice_edges: List[Partition], target: Node
) -> Dict[Partition, float]:
    """
    Compute depths for lattice edges with subset-based ordering.

    This function computes depths that respect subset relationships:
    - Subsets get smaller depth values (processed first)
    - Tree depth is used as a secondary ordering criterion
    - Missing edges get depth 0

    The ordering ensures {A} comes before {A,B} and {E} comes before {D,E}.

    Depth calculation formula:
    final_depth = (superset_of_count * 1000) + (partition_size * 10) + tree_depth

    This ensures:
    1. Subsets are processed before their supersets (superset_of_count factor).
    2. Smaller partitions are processed before larger ones (size factor).
    3. Tree depth provides fine-grained ordering.
    """
    lattice_edge_sort_map: Dict[Partition, float] = {}
    missing_edges: List[Partition] = []

    # First pass: collect basic tree depths
    basic_depths: Dict[Partition, float] = {}
    for edge in lattice_edges:
        node: Node | None = target.find_node_by_split(edge)
        if node and node.depth is not None:
            basic_depths[edge] = node.depth
        else:
            basic_depths[edge] = 0
            missing_edges.append(edge)

    # Second pass: compute subset-aware depths
    for edge in lattice_edges:
        tree_depth: float = basic_depths[edge]
        partition_size = len(edge)
        # Count how many other edges are proper subsets of this one.
        superset_of_count = _count_proper_subsets(edge, lattice_edges)

        # Compute final depth with subset-based ordering.
        # Partitions that are supersets of others get a higher depth value,
        # ensuring they are processed later.
        superset_depth = superset_of_count * 1000  # Primary ordering factor
        size_depth = partition_size * 10  # Secondary ordering factor

        final_depth = superset_depth + size_depth + tree_depth
        lattice_edge_sort_map[edge] = final_depth

    return lattice_edge_sort_map