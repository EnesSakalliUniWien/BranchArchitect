# Public API
"""
Tree interpolation module for creating smooth animations between phylogenetic trees.

This module provides the main public API for tree interpolation, creating
intermediate states that allow continuous morphing from one tree topology to another.
"""

from __future__ import annotations
from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.consensus_tree.consensus_tree import (
    calculate_consensus_tree,
)
from brancharchitect.tree_interpolation.consensus_tree.intermediate_tree import (
    calculate_intermediate_tree,
)
from brancharchitect.tree_interpolation.classical_interpolation import (
    classical_interpolation,
)
from brancharchitect.tree import Node
import logging

logger = logging.getLogger(__name__)


def interpolate_tree(target: Node, reference: Node) -> tuple[Node, Node, Node, Node]:
    """
    Interpolate between two trees to create intermediate and consensus trees.

    Returns a tuple of 4 trees:
    1. Intermediate tree from target (branch lengths averaged toward reference)
    2. Consensus from target (keeping only splits that are also in reference)
    3. Consensus from reference (keeping only splits that are also in target)
    4. Intermediate tree from reference (branch lengths averaged toward target)
    """
    target_splits: Dict[Partition, float] = target.to_weighted_splits()
    reference_splits: Dict[Partition, float] = reference.to_weighted_splits()

    intermediate_from_target: Node = calculate_intermediate_tree(
        target, reference_splits
    )
    intermediate_from_reference = calculate_intermediate_tree(reference, target_splits)

    consensus_from_target: Node = calculate_consensus_tree(
        intermediate_from_target, reference_splits
    )
    consensus_from_reference: Node = calculate_consensus_tree(
        intermediate_from_reference, target_splits
    )

    return (
        intermediate_from_target,
        consensus_from_target,
        consensus_from_reference,
        intermediate_from_reference,
    )


def interpolate_adjacent_tree_pairs(tree_list: List[Node]) -> List[Node]:
    """Interpolate between all adjacent pairs in a list of trees."""
    if len(tree_list) < 2:
        raise ValueError("Need at least 2 trees for interpolation")

    results: List[Node] = []
    for i in range(len(tree_list) - 1):
        target = tree_list[i]
        reference = tree_list[i + 1]

        trees = interpolate_tree(target, reference)
        results.append(target)
        results.extend(trees)

    results.append(tree_list[-1])
    return results


def create_classical_interpolation_fallback(
    current_state: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    s_edge: Partition,
    num_steps: int = 5,
) -> List[Node]:
    """
    Create a classical interpolation fallback when s-edge processing fails.

    Uses classical_interpolation to bridge from current state to reference tree.

    Args:
        current_state: The last successful interpolation state
        reference_tree: The target reference tree
        reference_weights: The reference tree weights
        s_edge: The s-edge that caused the failure (for logging)
        num_steps: Number of interpolation steps to generate

    Returns:
        A list of trees using classical interpolation
    """
    logger.info(f"Creating classical interpolation fallback for s-edge {s_edge}")

    try:
        # Calculate split data for classical interpolation
        current_splits: Dict[Partition, float] = current_state.to_weighted_splits()
        split_data: tuple[Dict[Partition, float], Dict[Partition, float]] = (
            current_splits,
            reference_weights,
        )

        # Use classical interpolation between current state and reference
        fallback_trees: List[Node] = classical_interpolation(
            current_state, reference_tree, split_data
        )

        # Ensure 5 trees are returned for consistency
        while len(fallback_trees) < num_steps:
            fallback_trees.append(fallback_trees[-1].deep_copy())
        return fallback_trees
    except Exception as e:
        logger.warning(f"Classical interpolation fallback failed: {e}")
        # Last resort: just return copies of current state
        return [current_state.deep_copy() for _ in range(num_steps)]
