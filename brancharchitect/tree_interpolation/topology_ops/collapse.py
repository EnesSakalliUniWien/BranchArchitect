"""
Collapse Operations - Removing Splits from Trees

This module handles topology collapse by removing splits from trees.
Use this when you need to remove internal nodes (reduce tree resolution).

Public API:
    - calculate_consensus_tree: Build consensus tree keeping only supported splits
    - collapse_zero_length_branches_for_node: Remove zero-length internal branches
    - execute_collapse_path: Remove a list of splits from a tree
    - execute_path: Combined collapse-then-expand transformation
    - create_collapsed_consensus_tree: Create collapsed tree under a pivot edge

Related modules:
    - expand.py: For adding splits (expanding topology)
    - weights.py: For manipulating branch lengths
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node

__all__ = [
    "calculate_consensus_tree",
    "collapse_zero_length_branches_for_node",
    "execute_collapse_path",
    "execute_path",
    "create_collapsed_consensus_tree",
]

logger: logging.Logger = logging.getLogger(__name__)


def calculate_consensus_tree(tree: Node, split_dict: Dict[Partition, float]) -> Node:
    """Calculate consensus tree based on split dictionary."""
    consensus_tree: Node = tree.deep_copy()
    return _calculate_consensus_tree(consensus_tree, split_dict)


def _collapse_iterative(
    root: Node,
    tol: float,
    destination_splits: Optional[PartitionSet[Partition]],
) -> None:
    """
    Collapse zero-length branches using iterative post-order traversal.

    This avoids recursion limits and function call overhead.
    Uses two stacks to generate post-order processing sequence.

    Args:
        root: Root node to process
        tol: Tolerance for considering a branch as zero-length
        destination_splits: Set of splits to preserve
    """
    # 1. Generate post-order traversal sequence
    # stack1 is for traversal, stack2 accumulates nodes in reverse post-order
    stack1: List[Node] = [root]
    stack2: List[Node] = []

    while stack1:
        node = stack1.pop()
        stack2.append(node)
        # Push children to stack1
        for child in node.children:
            stack1.append(child)

    # 2. Process nodes in post-order (popping from stack2)
    while stack2:
        cur = stack2.pop()

        if not cur.children:
            continue

        new_children: List[Node] = []
        local_change = False

        for ch in cur.children:
            ch_len = 0.0 if ch.length is None else float(ch.length)
            is_zero_length = ch_len <= tol and ch.children

            if not is_zero_length:
                new_children.append(ch)
                continue

            # Check if this split should be preserved
            should_preserve = (
                destination_splits is not None
                and ch.split_indices in destination_splits
            )

            if should_preserve:
                logger.debug(
                    f"[CONSENSUS COLLAPSE] Preserving zero-length branch "
                    f"(split exists in destination): {ch.split_indices}"
                )
                new_children.append(ch)
            else:
                # Splice grandchildren into current level
                # Note: grandchildren are already processed because of post-order
                for g in ch.children:
                    g.parent = cur
                    new_children.append(g)
                local_change = True

        if local_change:
            cur.children = new_children


def collapse_zero_length_branches_for_node(
    node: Node, tol: float = 0.0, destination_tree: Node | None = None
) -> None:
    """
    Collapse branches whose length is effectively zero (<= tol) in `node`'s subtree.
    Runs to a fixed point and refreshes indices/caches after structural changes.

    If destination_tree is provided, only collapse branches whose splits DON'T exist
    in the destination tree. This ensures we preserve topology needed for the final tree.
    """
    # Pre-compute destination splits if provided
    destination_splits: Optional[PartitionSet[Partition]] = None
    if destination_tree is not None:
        # Use PartitionSet for faster lookups (bitmask-based)
        destination_splits = PartitionSet(
            set(destination_tree.to_splits()), encoding=destination_tree.taxa_encoding
        )

    # Single pass iterative post-order traversal handles all collapses
    _collapse_iterative(node, tol, destination_splits)

    # Rebuild split indices & caches once after topology edits
    root = node.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches(propagate_up=True)


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


def execute_collapse_path(
    tree: Node,
    collapse_path: List[Partition],
    destination_tree: Optional[Node] = None,
) -> Node:
    """
    Execute collapse path by setting splits to zero length then collapsing.

    This function:
    1. Sets all collapse splits to zero length
    2. Collapses zero-length branches in one pass
    3. Preserves splits that exist in destination_tree
    4. Refreshes split indices once after completion

    Args:
        tree: The tree to modify (will be mutated)
        collapse_path: Splits to collapse
        destination_tree: If provided, preserve splits that exist here

    Returns:
        The modified tree with collapse splits removed

    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    if not collapse_path:
        return tree

    # Step 1: Set all collapse splits to zero length
    collapse_set = set(collapse_path)
    for split in collapse_set:
        node = tree.find_node_by_split(split)
        if node is not None:
            node.length = 0.0
            logger.debug(f"[COLLAPSE PATH] Set split {split.indices} to zero length")

    # Step 2: Collapse zero-length branches, preserving destination splits
    collapse_zero_length_branches_for_node(
        tree, tol=0.0, destination_tree=destination_tree
    )

    # Note: split indices and caches are refreshed inside collapse_zero_length_branches_for_node
    return tree


def execute_path(
    tree: Node,
    collapse_path: List[Partition],
    expand_path: List[Partition],
    destination_tree: Node,
) -> Node:
    """
    Execute complete collapse→expand path transformation.

    This function:
    1. Executes collapse_path first (using collapse functions)
    2. Executes expand_path second (using split_application)
    3. Applies reference weights from destination_tree
    4. Returns the modified tree

    Args:
        tree: The tree to modify (will be mutated)
        collapse_path: Splits to collapse
        expand_path: Splits to expand
        destination_tree: Reference for weights and preservation

    Returns:
        The modified tree with correct topology

    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    # Import here to avoid circular imports
    from brancharchitect.tree_interpolation.topology_ops.expand import (
        execute_expand_path,
    )

    # Step 1: Execute collapse path (preserving destination splits)
    execute_collapse_path(tree, collapse_path, destination_tree=destination_tree)

    # Step 2: Build reference weights from destination tree
    reference_weights: Dict[Partition, float] = {}
    for split in expand_path:
        dest_node = destination_tree.find_node_by_split(split)
        if dest_node is not None and dest_node.length is not None:
            reference_weights[split] = float(dest_node.length)

    # Step 3: Execute expand path with reference weights
    execute_expand_path(tree, expand_path, reference_weights)

    return tree


def create_collapsed_consensus_tree(
    down_phase_tree: Node,
    pivot_edge: Partition,
    tol: float = 0,  # small tolerance for numeric stability
    destination_tree: Node
    | None = None,  # to preserve splits that exist in destination
    copy: bool = True,  # whether to copy the tree first
) -> Node:
    """
    Collapse zero-length internal branches under pivot_edge, and refresh indices.

    If destination_tree is provided, only collapse zero-length branches whose splits DON'T
    exist in the destination. This ensures the final tree topology matches the destination.

    Args:
        down_phase_tree: The tree to collapse
        pivot_edge: The pivot edge partition
        tol: Tolerance for zero-length detection
        destination_tree: Optional destination tree for split preservation
        copy: If True, copy the tree first. If False, modify in place.
    """
    collapsed_tree: Node = down_phase_tree.deep_copy() if copy else down_phase_tree
    consensus_edge_node: Node | None = collapsed_tree.find_node_by_split(pivot_edge)
    if consensus_edge_node is not None:
        collapse_zero_length_branches_for_node(
            consensus_edge_node, tol=tol, destination_tree=destination_tree
        )
    # Note: cache refresh already done inside collapse_zero_length_branches_for_node
    return collapsed_tree
