"""
Core tree calculation functions for interpolation.

This module contains the fundamental algorithms for calculating
intermediate and consensus trees during the interpolation process.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.consensus.consensus_tree import apply_split_in_tree

logger: logging.Logger = logging.getLogger(__name__)


def calculate_intermediate_tree(tree: Node, split_dict: Dict[Partition, float]) -> Node:
    """Calculate intermediate tree with weighted splits using correct subset ordering."""
    intermediate_tree: Node = tree.deep_copy()
    original_split_dict: Dict[Partition, float] = tree.to_weighted_splits()
    _calculate_intermediate_ordered(intermediate_tree, split_dict, original_split_dict)
    return intermediate_tree


def calculate_intermediate(node: Node, split_dict: PartitionSet[Partition]) -> None:
    """
    Set branch lengths for a node and its subtree.

    - If node.split_indices in exact_keys: set to split_dict value (no averaging).
    - Else if node.split_indices in split_dict and original provided: average with original.
    - Else if node.split_indices in split_dict: set to split_dict value.
    - Else:
        - keep_missing=True: keep current node.length unchanged.
        - keep_missing=False: set to 0.0 (collapsible).
    """

    if node.split_indices not in split_dict:
        node.length = 0.0
    for child in node.children:
        calculate_intermediate(child, split_dict)


def calculate_intermediate_implicit(
    node: Node, to_be_set_zero: PartitionSet[Partition]
) -> Node:
    """
    Set branch lengths for a node and its subtree to zero if in to_be_set_to_zero.

    This is used for the down-phase where we set actionable zeros exactly.
    """
    _calculate_intermediate_implicit(node, to_be_set_zero)
    return node


def _calculate_intermediate_implicit(
    node: Node,
    to_be_set_to_zero: PartitionSet[Partition],
) -> None:
    if node.split_indices in to_be_set_to_zero:
        node.length = 0.0
    for child in node.children:
        _calculate_intermediate_implicit(child, to_be_set_to_zero)


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
    all_splits: list[Partition] = list(tree.to_splits())

    # Compute proper subset-based ordering
    depth_map: Dict[Partition, float] = compute_lattice_edge_depths(all_splits, tree)

    # Sort splits by depth (subsets first, ascending=True)
    sorted_splits: list[Partition] = sorted(all_splits, key=lambda p: depth_map[p])

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


def create_subtree_grafted_tree(
    base_tree: Node,
    ref_path_to_build: list[Partition],
) -> Node:
    """
    Create grafted tree with order-preserving split application.

    Args:
        base_tree: Tree to graft onto
        ref_path_to_build: Splits to apply
        reference_tree: Reference tree for ordering (optional)
        target_ordering_edge: Edge to use for final ordering (optional)
    """
    # Sort by partition size (number of taxa) in descending order
    # This ensures larger splits are applied before smaller ones
    sorted_ref_path = sorted(
        ref_path_to_build, key=lambda p: len(p.indices), reverse=True
    )

    grafted_tree = base_tree.deep_copy()
    for ref_split in sorted_ref_path:
        if ref_split not in base_tree.to_splits():
            # Apply the split to the base tree
            apply_split_in_tree(ref_split, grafted_tree)

    return grafted_tree
