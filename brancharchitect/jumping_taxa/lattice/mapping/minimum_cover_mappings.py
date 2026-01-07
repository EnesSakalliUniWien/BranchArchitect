"""
Solution element mapping using tree parent relationships.

Maps solution elements (moving subtrees) to their parent nodes in each tree,
providing a direct and accurate way to determine where subtrees are attached.
"""

from typing import Dict, List, Tuple, Optional

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
            # Try 1: Find exact node (monophyletic group)
            node_in_t1 = t1.find_node_by_split(solution)

            if node_in_t1 and node_in_t1.parent:
                # Ideally, map to the parent of the subtree root
                mapped_t1[edge][solution] = node_in_t1.parent.split_indices
            else:
                # Try 2: Find MRCA (scattered group)
                # MRCA is the 'container' node, so it acts like the parent
                mrca_t1 = _find_mrca_for_partition(t1, solution)
                if mrca_t1:
                    mapped_t1[edge][solution] = mrca_t1.split_indices
                else:
                    # Fallback: pivot edge
                    mapped_t1[edge][solution] = edge

            node_in_t2 = t2.find_node_by_split(solution)

            if node_in_t2 and node_in_t2.parent:
                mapped_t2[edge][solution] = node_in_t2.parent.split_indices
            else:
                mrca_t2 = _find_mrca_for_partition(t2, solution)
                if mrca_t2:
                    mapped_t2[edge][solution] = mrca_t2.split_indices
                else:
                    mapped_t2[edge][solution] = edge

    return mapped_t1, mapped_t2


def _find_mrca_for_partition(tree: Node, partition: Partition) -> Optional[Node]:
    """
    Find the Most Recent Common Ancestor (MRCA) for a set of taxa defined by a partition.

    Args:
        tree: The tree to search in.
        partition: The partition defining the taxa set.

    Returns:
        The MRCA Node if found, or None if the partition is empty or taxa not found.
    """
    if not partition.indices:
        return None
    # Identify leaf for each taxon index
    # We can iterate the tree's encoding or the partition's indices.
    # Partition indices are integers. We need to find the corresponding leaf nodes.
    # Since tree doesn't index leaves by int ID efficiently, we might need to look up by name.

    # Invert encoding for lookup: int -> name
    # (Optimization: could be cached, but this is fallback path)
    id_to_name = {v: k for k, v in tree.taxa_encoding.items()}

    first_leaf = None

    # Get the first leaf to start LCA traversal
    # We iterate indices to get names, then find split/node
    # Actually, we can use find_node_by_split(1 << idx) which is cached O(1) in the tree
    for idx in partition.indices:
        name = id_to_name.get(idx)
        if name:
            # Construct a single-taxon partition for lookup
            # This is efficient if the tree has build_split_index called (which find_node_by_split ensures)
            leaf_split = Partition.from_bitmask(1 << idx, tree.taxa_encoding)
            leaf_node = tree.find_node_by_split(leaf_split)

            if leaf_node:
                if first_leaf is None:
                    first_leaf = leaf_node
                else:
                    # Iteratively update LCA
                    # find_lowest_common_ancestor handles the traversal to root
                    first_leaf = first_leaf.find_lowest_common_ancestor(leaf_node)

                    # Optimization: If we hit the root, we can stop early
                    if first_leaf.parent is None:
                        return first_leaf

    return first_leaf
