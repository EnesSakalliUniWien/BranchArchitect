# Public API
"""
Tree interpolation module for creating smooth animations between phylogenetic trees.

This module provides the main public API for tree interpolation, creating
intermediate states that allow continuous morphing from one tree topology to another.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    calculate_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    calculate_intermediate_tree,
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


"""
Core tree calculation functions for interpolation.

This module contains the fundamental algorithms for calculating
intermediate and consensus trees during the interpolation process.
"""


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
