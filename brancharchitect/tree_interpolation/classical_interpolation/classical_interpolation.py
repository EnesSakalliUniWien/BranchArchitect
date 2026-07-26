# Public API
"""
Tree interpolation module for creating smooth animations between phylogenetic trees.

This module provides the main public API for tree interpolation, creating
intermediate states that allow continuous morphing from one tree topology to another.
"""

from __future__ import annotations
from typing import Dict, List
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


def interpolate_tree(source: Node, destination: Node) -> tuple[Node, Node, Node, Node]:
    """
    Interpolate between two trees to create intermediate and consensus trees.

    Returns a tuple of 4 trees:
    1. Intermediate tree from source (branch lengths averaged toward destination)
    2. Consensus from source (keeping only splits that are also in destination)
    3. Consensus from destination (keeping only splits that are also in source)
    4. Intermediate tree from destination (branch lengths averaged toward source)
    """
    source_splits: Dict[Partition, float] = source.to_weighted_splits()
    destination_splits: Dict[Partition, float] = destination.to_weighted_splits()

    intermediate_from_source: Node = calculate_intermediate_tree(
        source, destination_splits
    )
    intermediate_from_destination = calculate_intermediate_tree(
        destination, source_splits
    )

    consensus_from_source: Node = calculate_consensus_tree(
        intermediate_from_source, destination_splits
    )
    consensus_from_destination: Node = calculate_consensus_tree(
        intermediate_from_destination, source_splits
    )

    return (
        intermediate_from_source,
        consensus_from_source,
        consensus_from_destination,
        intermediate_from_destination,
    )


def interpolate_adjacent_tree_pairs(tree_list: List[Node]) -> List[Node]:
    """Interpolate between all adjacent pairs in a list of trees."""
    if len(tree_list) < 2:
        raise ValueError("Need at least 2 trees for interpolation")

    results: List[Node] = []
    for i in range(len(tree_list) - 1):
        source = tree_list[i]
        destination = tree_list[i + 1]

        trees = interpolate_tree(source, destination)
        results.append(source)
        results.extend(trees)

    results.append(tree_list[-1])
    return results
