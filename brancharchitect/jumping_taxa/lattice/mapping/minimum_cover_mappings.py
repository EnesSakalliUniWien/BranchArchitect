"""
Solution element mapping using tree parent relationships.

Maps solution elements (moving subtrees) to their parent nodes in each tree,
providing a direct and accurate way to determine where subtrees are attached.
"""

from typing import Dict, List, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node


def map_solution_elements_via_parent(
    pivot_edge_solutions: Dict[Partition, List[Partition]],
    t1: Node,
    t2: Node,
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Map solution elements using direct parent relationships in tree structure.

    For each solution element (moving subtree), find its parent in t1 and t2.
    The parent's split tells us where the subtree is attached.

    This is simpler and more accurate than overlap-based heuristics:
    - Uses actual tree topology instead of bitmask overlap
    - O(1) parent lookup vs O(n) overlap scanning
    - Directly answers "where is this subtree attached?"

    Args:
        pivot_edge_solutions: Dict mapping pivot edges to their solution partitions
        t1: First tree (source)
        t2: Second tree (destination)

    Returns:
        Two dictionaries (for t1 and t2) keyed by pivot edge, each mapping a
        solution partition -> parent partition (or the pivot edge as fallback).
    """
    mapped_t1: Dict[Partition, Dict[Partition, Partition]] = {}
    mapped_t2: Dict[Partition, Dict[Partition, Partition]] = {}

    for edge, solution_elements in pivot_edge_solutions.items():
        mapped_t1[edge] = {}
        mapped_t2[edge] = {}

        for solution in solution_elements:
            # Find the node with this split in each tree
            node_in_t1 = t1.find_node_by_split(solution)
            node_in_t2 = t2.find_node_by_split(solution)

            # Map to parent's split (or edge as fallback)
            if node_in_t1 and node_in_t1.parent:
                mapped_t1[edge][solution] = node_in_t1.parent.split_indices
            else:
                mapped_t1[edge][solution] = edge

            if node_in_t2 and node_in_t2.parent:
                mapped_t2[edge][solution] = node_in_t2.parent.split_indices
            else:
                mapped_t2[edge][solution] = edge

    return mapped_t1, mapped_t2
