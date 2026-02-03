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
from typing import Dict, Optional, List
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


def set_expand_splits_to_dest_weight(
    tree: Node,
    expand_splits: List[Partition],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Set expand splits (new in destination) to their destination weight.

    Expand splits don't exist in the source tree, so there's nothing to average -
    we simply set them to their final destination weight.

    Args:
        tree: The tree to modify (mutated in place)
        expand_splits: List of splits that are new in the destination
        destination_weights: Weights from the destination tree
    """
    for split in expand_splits:
        node = tree.find_node_by_split(split)
        if node is not None:
            node.length = destination_weights.get(split, 0.0)


def average_weights_under_pivot(
    tree: Node,
    pivot_node: Node,
    expand_set: set[Partition],
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Average branch weights for all splits under the pivot edge.

    For the first mover, we process ALL splits under the pivot:
    - Expand splits (new in destination): set to destination weight
    - Shared splits (in both trees): average (source + dest) / 2

    Args:
        tree: The tree to modify (mutated in place)
        pivot_node: The node representing the pivot edge
        expand_set: Set of splits that are expand (new) splits
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    splits_under_pivot = list(pivot_node.to_splits(with_leaves=True))

    for split in splits_under_pivot:
        node = tree.find_node_by_split(split)
        if node is None:
            continue

        if split in expand_set:
            # Expand split: set to destination weight
            node.length = destination_weights.get(split, 0.0)
        else:
            # Shared split: average (source + dest) / 2
            src_weight = (
                source_weights.get(split, node.length or 0.0)
                if source_weights
                else (node.length or 0.0)
            )
            dest_weight = destination_weights.get(split, 0.0)
            node.length = (src_weight + dest_weight) / 2.0


def finalize_branch_weights(
    tree: Node,
    current_pivot_edge: Partition,
    expand_path: List[Partition],
    is_first_mover: bool,
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Finalize branch weights during the snap phase.

    Weight application strategy:
    - First mover: Average all shared splits under pivot, set expand splits to dest weight
    - Subsequent movers: Only set their expand splits to destination weight
                         (shared splits were already averaged by first mover)

    Args:
        tree: The tree to modify (mutated in place)
        current_pivot_edge: The pivot edge being processed
        expand_path: List of splits to expand for this mover
        is_first_mover: Whether this is the first mover for this pivot
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    pivot_node = tree.find_node_by_split(current_pivot_edge)

    if is_first_mover:
        # First mover: Average all shared splits under pivot + pivot edge itself
        if pivot_node is not None and source_weights is not None:
            expand_set = set(expand_path)
            average_weights_under_pivot(
                tree, pivot_node, expand_set, source_weights, destination_weights
            )
    else:
        # Subsequent movers: Only set expand splits to destination weight
        # Shared splits were already averaged by the first mover
        set_expand_splits_to_dest_weight(tree, expand_path, destination_weights)
