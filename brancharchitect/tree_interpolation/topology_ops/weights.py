"""
Weight Operations - Branch Length Manipulation

This module handles branch length (weight) manipulation during interpolation.
Use this when you need to set, interpolate, or zero branch lengths.

Public API:
    - calculate_intermediate_tree: Create tree with interpolated branch lengths
    - apply_zero_branch_lengths: Set specified branches to zero length
    - apply_reference_weights_to_path: Copy branch lengths from a reference tree

Related modules:
    - expand.py: For adding splits (expanding topology)
    - collapse.py: For removing splits (collapsing topology)
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

from brancharchitect.elements.partition_set import PartitionSet

__all__ = [
    "calculate_intermediate_tree",
    "apply_zero_branch_lengths",
    "apply_reference_weights_to_path",
]

logger: logging.Logger = logging.getLogger(__name__)


def calculate_intermediate_tree(
    tree: Node, destination_splits: Dict[Partition, float]
) -> Node:
    """
    Calculate intermediate tree with weighted splits using correct subset ordering.

    Args:
        tree: The base tree to modify
        destination_splits: Dictionary of splits and their target weights

    Returns:
        The modified intermediate tree with zeroed or averaged branch lengths.
    """
    intermediate_tree: Node = tree.deep_copy()
    source_splits: Dict[Partition, float] = tree.to_weighted_splits()
    # Detect zero-only instruction set (all targets are 0.0)
    zero_only: bool = all((v == 0 or v == 0.0) for v in destination_splits.values())

    # We can use the keys from source_splits as the list of all splits
    # since we just calculated it.
    all_splits = list(source_splits.keys())

    if zero_only:
        _apply_zero_only_branch_lengths(
            intermediate_tree,
            destination_splits,
            source_splits,
            all_splits=all_splits,
        )
    else:
        _apply_interpolation_branch_lengths(
            intermediate_tree,
            destination_splits,
            source_splits,
            all_splits=all_splits,
        )

    return intermediate_tree


def apply_zero_branch_lengths(
    node: Node, splits_to_zero: PartitionSet[Partition]
) -> Node:
    """
    Set branch lengths for a node and its subtree to zero if in splits_to_zero.

    This is used for the down-phase where we set specific branches to zero length.
    """
    _apply_zero_branch_lengths_recursive(node, splits_to_zero)
    return node


def _apply_zero_branch_lengths_recursive(
    node: Node,
    splits_to_zero: PartitionSet[Partition],
) -> None:
    if node.split_indices in splits_to_zero:
        node.length = 0.0
    for child in node.children:
        _apply_zero_branch_lengths_recursive(child, splits_to_zero)


def _apply_zero_only_branch_lengths(
    tree: Node,
    destination_splits: Dict[Partition, float],
    source_splits: Optional[Dict[Partition, float]] = None,
    all_splits: Optional[list[Partition]] = None,
) -> None:
    """
    Apply zero-only logic: only set the specified destination splits to zero length.
    Preserves all other branch lengths from the source.
    """
    if all_splits is None:
        all_splits = list(tree.to_splits())

    for split in all_splits:
        node = tree.find_node_by_split(split)
        if node is not None:
            if split in destination_splits:
                # Only set the explicitly requested splits to zero
                node.length = 0.0
            elif source_splits is not None and split in source_splits:
                # Keep others unchanged from source
                node.length = source_splits[split]


def _apply_interpolation_branch_lengths(
    tree: Node,
    destination_splits: Dict[Partition, float],
    source_splits: Optional[Dict[Partition, float]] = None,
    all_splits: Optional[list[Partition]] = None,
) -> None:
    """
    Apply general interpolation logic: average weights and zero missing splits.
    """
    if all_splits is None:
        all_splits = list(tree.to_splits())

    for split in all_splits:
        node = tree.find_node_by_split(split)
        if node is not None:
            if split in destination_splits:
                target_val = destination_splits[split]
                if target_val == 0 or target_val == 0.0:
                    # Explicit zero forces exact zero-length
                    node.length = 0.0
                elif source_splits is not None and split in source_splits:
                    # Average with source if available
                    node_length = source_splits[split]
                    node.length = (target_val + node_length) / 2
                else:
                    node.length = target_val
            else:
                # Not present in destination set: set to zero
                node.length = 0.0


def apply_reference_weights_to_path(
    tree: Node,
    expand_path: list[Partition],
    reference_weights: Dict[Partition, float],
) -> None:
    """Set branch lengths on nodes along a path to match reference weights.

    Mutates the provided tree in place.
    """
    for ref_split in expand_path:
        node: Node | None = tree.find_node_by_split(ref_split)
        if node is not None:
            node.length = reference_weights.get(ref_split, 1)
