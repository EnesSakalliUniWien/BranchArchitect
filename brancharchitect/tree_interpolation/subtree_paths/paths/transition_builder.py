"""Transition path builder (pure data extraction).

Computes per-active-changing-split subtree paths on both the reference and
target trees. This module is pure: it does not mutate trees and only returns
path datasets for upstream orchestration.
"""

from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
import logging

logger = logging.getLogger(__name__)


def calculate_subtree_paths(
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    reference_tree: Node,
    target_tree: Node,
) -> tuple[
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
]:
    """
    Calculates the subtree paths for both reference and target trees for each
    current pivot edge.

    Args:
        jumping_subtree_solutions: A dictionary mapping current pivot edges
            to their subtree sets.
        reference_tree: The reference tree.
        target_tree: The target tree.

    Returns:
        A tuple containing two dictionaries:
        - reference_subtree_paths: Paths in the reference tree, keyed by
          current pivot edge and then subtree.
        - target_subtree_paths: Paths in the target tree, keyed by
          current pivot edge and then subtree.
    """
    reference_subtree_paths: Dict[
        Partition, Dict[Partition, PartitionSet[Partition]]
    ] = {}
    target_subtree_paths: Dict[Partition, Dict[Partition, PartitionSet[Partition]]] = {}

    for current_pivot_edge, subtrees in jumping_subtree_solutions.items():
        reference_subtree_paths[current_pivot_edge] = {}
        target_subtree_paths[current_pivot_edge] = {}

        for subtree in subtrees:
            reference_subtree_node_paths: List[Node] = (
                reference_tree.find_path_between_splits(subtree, current_pivot_edge)
            )

            target_subtree_node_paths: List[Node] = (
                target_tree.find_path_between_splits(subtree, current_pivot_edge)
            )

            # Extract partitions from nodes, excluding endpoints (subtree and split)
            reference_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in reference_subtree_node_paths}
            )
            target_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in target_subtree_node_paths}
            )

            # Remove endpoint partitions that shouldn't be in collapse/expand paths
            reference_partitions.discard(subtree)  # Remove subtree endpoint
            reference_partitions.discard(
                current_pivot_edge
            )  # Remove split endpoint
            target_partitions.discard(subtree)  # Remove subtree endpoint
            target_partitions.discard(current_pivot_edge)  # Remove split endpoint

            reference_subtree_paths[current_pivot_edge][subtree] = (
                reference_partitions
            )

            target_subtree_paths[current_pivot_edge][subtree] = target_partitions

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[paths] pivot=%s subtree=%s target_partitions=%s",
                    current_pivot_edge.bipartition(),
                    subtree.bipartition(),
                    [list(p.indices) for p in target_partitions],
                )

    return reference_subtree_paths, target_subtree_paths
