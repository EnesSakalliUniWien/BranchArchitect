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
    "apply_mean_weights_to_path",
    "compensate_patristic_distances",
    "compensate_tip_distances",  # Deprecated alias for backward compatibility
    "distribute_path_weights",
    "get_virtual_collapse_weights",
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


def apply_mean_weights_to_path(
    tree: Node,
    splits_to_update: list[Partition],
    destination_weights: Dict[Partition, float],
    expand_splits: Optional[PartitionSet[Partition]] = None,
    source_weights: Optional[Dict[Partition, float]] = None,
) -> None:
    """Apply mean interpolation (L1 + L2) / 2 for splits.

    For common splits (exist in both source and destination):
        new_length = (source_length + destination_length) / 2

    For expand splits (new in destination):
        new_length = (0.0 + destination_length) / 2
        This treats the split as having length 0 in the source.

    Args:
        tree: The tree to modify (mutated in place)
        splits_to_update: List of splits to update
        destination_weights: Target weights from destination tree
        expand_splits: Set of splits that were newly created (expand path).
        source_weights: Optional dictionary of original source weights.
    """
    expand_set = (
        expand_splits if expand_splits is not None else PartitionSet(encoding={})
    )

    for split in splits_to_update:
        node: Node | None = tree.find_node_by_split(split)
        if node is not None:
            dest_weight = destination_weights.get(split, node.length)

            if split in expand_set:
                # Expand splits: Interpolate from 0 (Source) to Dest
                # Mean = (0.0 + Dest) / 2.0
                node.length = dest_weight / 2.0
            else:
                # Common splits: interpolate with mean
                if source_weights is not None and split in source_weights:
                    current_weight = source_weights[split]
                else:
                    current_weight = node.length

                node.length = (current_weight + dest_weight) / 2.0


def compensate_patristic_distances(
    current_tree: Node,
    source_tree: Node,
    destination_tree: Node,
    progress: float = 0.5,
    leaves_to_process: list[Node] | None = None,
    _debug_stats: dict | None = None,
) -> None:
    """
    Adjust pendant edge lengths to achieve target patristic distances.

    Computes target patristic distance for each leaf as a linear interpolation
    between source and destination distances based on progress fraction:
        D_target = (1 - progress) * D_source + progress * D_dest

    This compensates for the "shrinking" effect of BHV geodesic interpolation
    where internal branches collapse, ensuring the tree visual remains full-sized.

    The term "patristic distance" is the standard phylogenetic term for the sum
    of branch lengths along the path between two nodes (here: root to tip).

    Args:
        current_tree: The tree being interpolated (modified in place)
        source_tree: The original source tree (reference for start distances)
        destination_tree: The target tree (reference for end distances)
        progress: Interpolation progress fraction (0.0 = source, 1.0 = destination).
                  Default 0.5 for backward compatibility with midpoint averaging.
        leaves_to_process: Optional list of leaf nodes to compensate. If None,
                           all leaves in current_tree are processed. Use this to
                           limit compensation to leaves under a specific pivot edge.
        _debug_stats: Optional dict to collect debug statistics

    Raises:
        ValueError: If progress is not in [0.0, 1.0]
    """
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0.0, 1.0], got {progress}")

    # Calculate reference patristic distances
    source_dists = _get_patristic_distances(source_tree)
    dest_dists = _get_patristic_distances(destination_tree)

    # Use provided leaves or all leaves
    leaves = (
        leaves_to_process
        if leaves_to_process is not None
        else current_tree.get_leaves()
    )

    # Apply compensation to each leaf
    for leaf in leaves:
        name = leaf.name
        if name not in source_dists or name not in dest_dists:
            continue

        # Linear interpolation of target distance based on progress
        target_dist = (1.0 - progress) * source_dists[name] + progress * dest_dists[
            name
        ]

        # Calculate current actual distance
        current_dist = _calculate_root_to_tip_distance(leaf)

        # Calculate deficit and apply to pendant edge
        delta = target_dist - current_dist

        if leaf.length is None:
            leaf.length = 0.0

        original_length = leaf.length
        leaf.length += delta

        # Clamp to non-negative
        if leaf.length < 0:
            # Collect debug stats if provided
            if _debug_stats is not None:
                _debug_stats["negative_count"] = (
                    _debug_stats.get("negative_count", 0) + 1
                )
                if len(_debug_stats.get("samples", [])) < 20:
                    _debug_stats.setdefault("samples", []).append(
                        {
                            "name": name,
                            "source_dist": source_dists[name],
                            "dest_dist": dest_dists[name],
                            "current_dist": current_dist,
                            "target_dist": target_dist,
                            "delta": delta,
                            "original_length": original_length,
                            "progress": progress,
                        }
                    )
            logger.warning(
                f"Pendant edge for {name} clamped from {leaf.length:.4f} to 0"
            )
            leaf.length = 0.0


def _get_patristic_distances(tree: Node) -> Dict[str, float]:
    """Calculate root-to-tip (patristic) distances for all leaves.

    Args:
        tree: The tree to calculate distances for

    Returns:
        Dictionary mapping leaf names to their root-to-tip distances
    """
    distances: Dict[str, float] = {}
    for leaf in tree.get_leaves():
        distances[leaf.name] = _calculate_root_to_tip_distance(leaf)
    return distances


def _calculate_root_to_tip_distance(leaf: Node) -> float:
    """Sum branch lengths from leaf to root.

    Args:
        leaf: The leaf node to calculate distance for

    Returns:
        Total branch length from leaf to root
    """
    total = 0.0
    current: Node | None = leaf
    while current is not None:
        if current.length is not None:
            total += current.length
        current = current.parent
    return total


def compensate_tip_distances(
    current_tree: Node,
    source_tree: Node,
    destination_tree: Node,
) -> None:
    """
    Deprecated: Use compensate_patristic_distances instead.

    This function is kept for backward compatibility and calls
    compensate_patristic_distances with progress=0.5 (midpoint).

    Args:
        current_tree: The tree being interpolated (modified in place)
        source_tree: The original source tree (reference for start distances)
        destination_tree: The target tree (reference for end distances)
    """
    import warnings

    warnings.warn(
        "compensate_tip_distances is deprecated, use compensate_patristic_distances instead",
        DeprecationWarning,
        stacklevel=2,
    )
    compensate_patristic_distances(
        current_tree=current_tree,
        source_tree=source_tree,
        destination_tree=destination_tree,
        progress=0.5,
    )


def distribute_path_weights(
    tree: Node, path: list[Partition], operation: str = "add"
) -> None:
    """
    Distribute the mass of internal path branches to their descendant leaves.

    Args:
        tree: The tree to modify (mutated in place).
        path: List of internal Partitions whose weight should be moved to pendants.
        operation: "add" to move mass to pendants (Collapse phase),
                   "subtract" to pull mass from pendants (Expand phase).
    """
    if not path:
        return

    leaves = tree.get_leaves()
    # Path might contain internal splits that exist in the tree.
    # We sum their lengths and move them to leaves strictly underneath.
    for split in path:
        node = tree.find_node_by_split(split)
        if node is None or node.length == 0:
            continue

        L = node.length
        # Find all leaves strictly under this split using bitmask logic
        split_bits = split.bitmask

        for leaf in leaves:
            leaf_bits = leaf.split_indices.bitmask
            if (leaf_bits & split_bits) == leaf_bits:
                if operation == "add":
                    leaf.length += L
                else:
                    leaf.length -= L
                    if leaf.length < 0:
                        logger.warning(
                            f"Pendant edge for {leaf.name} clamped from {leaf.length:.4f} to 0 during mass distribution"
                        )
                        leaf.length = 0.0


def get_virtual_collapse_weights(
    tree: Node, path: list[Partition]
) -> Dict[Partition, float]:
    """
    Calculate the 'Virtual' weights of a tree if the specified path was collapsed.

    The internal branches in 'path' will have weight 0 in the returned dict,
    and their weights will be added to the weights of their descendant leaves.
    """
    # Start with true weighted splits (includes leaves)
    weights = tree.to_weighted_splits()

    leaves = tree.get_leaves()

    for split in path:
        if split not in weights:
            # If split is not in destinaton tree, it already has 0 weight effectively
            continue

        L = weights[split]
        split_bits = split.bitmask

        # Add mass to descendants in the weighting dict
        for leaf in leaves:
            leaf_bits = leaf.split_indices.bitmask
            if (leaf_bits & split_bits) == leaf_bits:
                leaf_split = leaf.split_indices
                weights[leaf_split] = weights.get(leaf_split, 0.0) + L

        # Zero out the collapsed internal branch weight in the virtual map
        weights[split] = 0.0

    return weights
