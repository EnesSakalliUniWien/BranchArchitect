"""Discover collapse and expand paths for active-split interpolation."""

from __future__ import annotations

from typing import Dict, List

from brancharchitect.elements.partition import Partition, partition_size_bitmask_key
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node

PivotSubtreePaths = Dict[Partition, Dict[Partition, PartitionSet[Partition]]]


def find_unassigned_source_splits_under_pivot(
    current_pivot_edge: Partition,
    mover_subtrees: List[Partition],
    source_only_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """
    Find source-only splits under a pivot that are not on any mover path.

    These splits still need to collapse for the pivot transition to be
    complete, even though no selected mover owns them directly.
    """
    mover_indices: set[int] = set()
    for subtree in mover_subtrees:
        mover_indices.update(subtree.indices)

    pivot_indices = set(current_pivot_edge.indices)
    residual_splits: PartitionSet[Partition] = PartitionSet(
        encoding=source_only_splits.encoding
    )

    for split in source_only_splits:
        split_indices = set(split.indices)
        if not split_indices.issubset(pivot_indices):
            continue
        if split_indices == pivot_indices:
            continue
        if split_indices.isdisjoint(mover_indices):
            residual_splits.add(split)

    return residual_splits


def build_pivot_subtree_transition_paths(
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    destination_tree: Node,
    source_tree: Node,
) -> tuple[PivotSubtreePaths, PivotSubtreePaths]:
    """
    Build destination expand paths and source collapse paths for each pivot.

    Returns:
        A tuple of `(destination_subtree_paths, source_subtree_paths)`, each
        keyed by pivot edge and then mover subtree.
    """
    destination_subtree_paths: PivotSubtreePaths = {}
    source_subtree_paths: PivotSubtreePaths = {}

    source_splits: PartitionSet[Partition] = source_tree.to_splits()
    destination_splits: PartitionSet[Partition] = destination_tree.to_splits()
    source_only_splits = source_splits - destination_splits
    destination_only_splits = destination_splits - source_splits

    for current_pivot_edge, subtrees in sorted(
        jumping_subtree_solutions.items(),
        key=lambda item: partition_size_bitmask_key(item[0]),
    ):
        destination_subtree_paths[current_pivot_edge] = {}
        source_subtree_paths[current_pivot_edge] = {}

        residual_splits = find_unassigned_source_splits_under_pivot(
            current_pivot_edge, subtrees, source_only_splits
        )

        for subtree in sorted(subtrees, key=partition_size_bitmask_key):
            destination_node_path: List[Node] = (
                destination_tree.find_path_between_splits(subtree, current_pivot_edge)
            )
            source_node_path: List[Node] = source_tree.find_path_between_splits(
                subtree, current_pivot_edge
            )

            destination_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in destination_node_path}
            )
            source_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in source_node_path}
            )

            source_partitions = source_partitions.intersection(source_only_splits)
            destination_partitions = destination_partitions.intersection(
                destination_only_splits
            )

            destination_partitions.discard(current_pivot_edge)
            source_partitions.discard(current_pivot_edge)
            source_partitions.discard(subtree)

            if subtree in source_splits:
                destination_partitions.discard(subtree)

            destination_subtree_paths[current_pivot_edge][
                subtree
            ] = destination_partitions
            source_subtree_paths[current_pivot_edge][subtree] = source_partitions

        if residual_splits and source_subtree_paths[current_pivot_edge]:
            first_subtree = min(
                source_subtree_paths[current_pivot_edge].keys(),
                key=lambda partition: partition.bitmask,
            )
            source_subtree_paths[current_pivot_edge][first_subtree] = (
                source_subtree_paths[current_pivot_edge][first_subtree]
                | residual_splits
            )

    return destination_subtree_paths, source_subtree_paths
