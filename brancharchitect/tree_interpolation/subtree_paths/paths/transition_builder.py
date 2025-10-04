"""Transition path builder (pure data extraction).

Computes per–active-changing-split subtree paths on both the reference and
target trees. This module is pure: it does not mutate trees and only returns
path datasets for upstream orchestration.
"""

from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node


def calculate_subtree_paths(
    jumping_subtree_solutions: Dict[Partition, List[List[Partition]]],
    reference_tree: Node,
    target_tree: Node,
) -> tuple[
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
]:
    """
    Calculates the subtree paths for both reference and target trees for each
    active-changing split.

    Args:
        jumping_subtree_solutions: A dictionary mapping active-changing splits
            to their subtree sets.
        reference_tree: The reference tree.
        target_tree: The target tree.

    Returns:
        A tuple containing two dictionaries:
        - reference_subtree_paths: Paths in the reference tree, keyed by
          active-changing split and then subtree.
        - target_subtree_paths: Paths in the target tree, keyed by
          active-changing split and then subtree.
    """
    reference_subtree_paths: Dict[
        Partition, Dict[Partition, PartitionSet[Partition]]
    ] = {}
    target_subtree_paths: Dict[Partition, Dict[Partition, PartitionSet[Partition]]] = {}

    for active_changing_split, split_subtrees in jumping_subtree_solutions.items():
        reference_subtree_paths[active_changing_split] = {}
        target_subtree_paths[active_changing_split] = {}

        for subtree_set in split_subtrees:
            for subtree in subtree_set:
                reference_subtree_node_paths: List[Node] = (
                    reference_tree.find_path_between_splits(
                        subtree, active_changing_split
                    )
                )

                target_subtree_node_paths: List[Node] = (
                    target_tree.find_path_between_splits(subtree, active_changing_split)
                )

                # Extract partitions from nodes, excluding endpoints (subtree and split)
                reference_partitions = set(
                    [node.split_indices for node in reference_subtree_node_paths]
                )
                target_partitions = set(
                    [node.split_indices for node in target_subtree_node_paths]
                )

                # Remove endpoint partitions that shouldn't be in collapse/expand paths
                reference_partitions.discard(subtree)  # Remove subtree endpoint
                reference_partitions.discard(
                    active_changing_split
                )  # Remove split endpoint
                target_partitions.discard(subtree)  # Remove subtree endpoint
                target_partitions.discard(
                    active_changing_split
                )  # Remove split endpoint

                reference_path_as_partitions: PartitionSet[Partition] = PartitionSet(
                    reference_partitions
                )
                target_path_as_partitions: PartitionSet[Partition] = PartitionSet(
                    target_partitions
                )

                reference_subtree_paths[active_changing_split][subtree] = (
                    reference_path_as_partitions
                )

                target_subtree_paths[active_changing_split][subtree] = (
                    target_path_as_partitions
                )

    return reference_subtree_paths, target_subtree_paths
