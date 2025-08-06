"""
Core tree calculation functions for interpolation.

This module contains the fundamental algorithms for calculating
intermediate and consensus trees during the interpolation process.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.ordering import compute_lattice_edge_depths


def calculate_intermediate_tree(tree: Node, split_dict: Dict[Partition, float]) -> Node:
    """Calculate intermediate tree with weighted splits using correct subset ordering."""
    intermediate_tree = tree.deep_copy()
    original_split_dict = tree.to_weighted_splits()

    # CRITICAL FIX: Process splits in subset-to-superset order
    _calculate_intermediate_ordered(intermediate_tree, split_dict, original_split_dict)
    return intermediate_tree


def calculate_intermediate(
    node: Node,
    split_dict: Dict[Partition, float],
    original_split_dict: Optional[Dict[Partition, float]] = None,
) -> None:
    """
    Calculate intermediate branch lengths for a node and its children.

    For each split:
    - If split exists in target: average the branch lengths
    - If split missing in target: set length to 0 (will be collapsed)
    """
    if node.split_indices not in split_dict:
        node.length = 0
    else:
        if original_split_dict is not None:
            node_length: float = original_split_dict[node.split_indices]
            node.length = (split_dict[node.split_indices] + node_length) / 2
        else:
            node.length = split_dict[node.split_indices]
    for child in node.children:
        calculate_intermediate(child, split_dict, original_split_dict)


def calculate_consensus_tree(tree: Node, split_dict: Dict[Partition, float]) -> Node:
    """Calculate consensus tree based on split dictionary."""
    consensus_tree = tree.deep_copy()
    return _calculate_consensus_tree(consensus_tree, split_dict)


def _calculate_intermediate_ordered(
    tree: Node,
    split_dict: Dict[Partition, float],
    original_split_dict: Optional[Dict[Partition, float]] = None,
) -> None:
    """
    Calculate intermediate branch lengths using correct subset-to-superset ordering.

    This is the CRITICAL FIX for the tree interpolation algorithm:
    - Process splits from smallest subsets to largest subsets
    - Ensures that smaller internal nodes are handled before their containers
    - Prevents incorrect branch length inheritance in nested structures
    """

    # Get all splits in the tree
    all_splits = list(tree.to_splits())

    # Compute proper subset-based ordering
    depth_map = compute_lattice_edge_depths(all_splits, tree)

    # Sort splits by depth (subsets first, ascending=True)
    sorted_splits = sorted(all_splits, key=lambda p: depth_map[p])

    # Process each split in correct order
    for split in sorted_splits:
        node = tree.find_node_by_split(split)
        if node is not None:
            if split not in split_dict:
                # Set to zero if split not in target
                node.length = 0
            else:
                # Average with original if available
                if original_split_dict is not None and split in original_split_dict:
                    node_length = original_split_dict[split]
                    node.length = (split_dict[split] + node_length) / 2
                else:
                    node.length = split_dict[split]


def collapse_zero_length_branches_for_node(node: Node, tol: float = 1e-12) -> None:
    """
    Collapse branches whose length is effectively zero (<= tol) in `node`'s subtree.
    Runs to a fixpoint and refreshes indices/caches after structural changes.
    """

    def _collapse_once(cur: Node) -> bool:
        changed = False
        if not cur.children:
            return False

        new_children: List[Node] = []
        for ch in cur.children:
            ch_len = 0.0 if ch.length is None else float(ch.length)
            if ch.children and ch_len <= tol:
                # splice grandchildren
                for g in ch.children:
                    g.parent = cur
                    new_children.append(g)
                changed = True
            else:
                new_children.append(ch)

        # Only assign if something actually changed to keep references stable
        if changed:
            cur.children = new_children

        # Recurse; keep OR-ing change flags so we know if another outer pass is needed
        for ch in cur.children:
            changed = _collapse_once(ch) or changed
        return changed

    # Fixpoint: keep collapsing until no structural change occurs
    while _collapse_once(node):
        pass

    # IMPORTANT: rebuild split indices & caches after topology edits
    root = node.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches(propagate_up=True)

    # Recursively process all children
    for child in node.children:
        collapse_zero_length_branches_for_node(child)


def _calculate_consensus_tree(node: Node, split_dict: Dict[Partition, float]) -> Node:
    """Recursively build consensus tree by keeping only supported splits."""
    if not node.children:
        return node

    new_children: List[Node] = []
    for child in node.children:
        processed_child = _calculate_consensus_tree(child, split_dict)
        if processed_child.children:
            if processed_child.split_indices in split_dict:
                new_children.append(processed_child)
            else:
                for grandchild in processed_child.children:
                    grandchild.parent = node
                    new_children.append(grandchild)
        else:
            new_children.append(processed_child)

    node.children = new_children
    for child in node.children:
        child.parent = node
    return node


def classical_interpolation(
    target: Node,
    reference: Node,
    split_data: Tuple[Dict[Partition, float], Dict[Partition, float]],
) -> List[Node]:
    """Create consensus tree sequence and mappings."""
    split_dict1, split_dict2 = split_data

    # Create intermediate and consensus trees
    it1: Node = calculate_intermediate_tree(target, split_dict2)
    it2: Node = calculate_intermediate_tree(reference, split_dict1)
    c1: Node = calculate_consensus_tree(it1, split_dict2)
    c2: Node = calculate_consensus_tree(it2, split_dict1)

    return [it1, c1, c2, it2]
