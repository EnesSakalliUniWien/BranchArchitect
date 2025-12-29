"""Transition path builder (pure data extraction).

Computes per-active-changing-split subtree paths on both the destination and
source trees. This module is pure: it does not mutate trees and only returns
path datasets for upstream orchestration.
"""

from typing import Dict, List, Set
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node


def calculate_subtree_paths(
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    destination_tree: Node,
    source_tree: Node,
) -> tuple[
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
]:
    """
    Calculates the subtree paths for both destination and source trees for each
    current pivot edge.

    Args:
        jumping_subtree_solutions: A dictionary mapping current pivot edges
            to their subtree sets.
        destination_tree: The destination tree (expand paths - splits to create).
        source_tree: The source tree (collapse paths - splits to remove).

    Returns:
        A tuple containing two dictionaries:
        - destination_subtree_paths: Paths in the destination tree, keyed by
          current pivot edge and then subtree. Used as expand paths.
        - source_subtree_paths: Paths in the source tree, keyed by
          current pivot edge and then subtree. Used as collapse paths.
    """
    destination_subtree_paths: Dict[
        Partition, Dict[Partition, PartitionSet[Partition]]
    ] = {}
    source_subtree_paths: Dict[Partition, Dict[Partition, PartitionSet[Partition]]] = {}

    # Pre-compute splits in source tree for existence checks
    source_splits: Set[Partition] = set(source_tree.to_splits())

    for current_pivot_edge, subtrees in jumping_subtree_solutions.items():
        destination_subtree_paths[current_pivot_edge] = {}
        source_subtree_paths[current_pivot_edge] = {}

        for subtree in subtrees:
            destination_node_paths: List[Node] = (
                destination_tree.find_path_between_splits(subtree, current_pivot_edge)
            )

            source_node_paths: List[Node] = source_tree.find_path_between_splits(
                subtree, current_pivot_edge
            )

            # Extract partitions from nodes
            destination_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in destination_node_paths}
            )
            source_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in source_node_paths}
            )

            # Always remove pivot edge endpoint from both paths
            destination_partitions.discard(current_pivot_edge)
            source_partitions.discard(current_pivot_edge)

            # Always remove subtree from collapse path (source)
            source_partitions.discard(subtree)

            # Only remove subtree from expand path (destination) if it
            # already exists in source - otherwise it needs to be created
            if subtree in source_splits:
                destination_partitions.discard(subtree)

            destination_subtree_paths[current_pivot_edge][subtree] = (
                destination_partitions
            )
            source_subtree_paths[current_pivot_edge][subtree] = source_partitions

    return destination_subtree_paths, source_subtree_paths
