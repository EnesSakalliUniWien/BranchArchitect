from __future__ import annotations

from typing import Dict, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

MISSING_VISUAL_ORDER = 10**12


def taxa_for_partitions(partitions: list[Partition]) -> set[str]:
    taxa: set[str] = set()
    for partition in partitions:
        taxa.update(partition.taxa)
    return taxa


def expand_paths_require_leaf_order_change(
    expand_paths: list[Partition],
    leaf_order: list[str],
) -> bool:
    position_by_taxon = {taxon: index for index, taxon in enumerate(leaf_order)}

    for partition in expand_paths:
        positions = sorted(
            position_by_taxon[taxon]
            for taxon in partition.taxa
            if taxon in position_by_taxon
        )
        if len(positions) <= 1:
            continue
        if positions[-1] - positions[0] + 1 != len(positions):
            return True

    return False


def build_destination_mover_order_key(
    destination_tree: Node,
    current_pivot_edge: Partition,
    mover_partitions: set[Partition],
) -> Dict[Partition, Tuple[int, ...]]:
    destination_subtree = destination_tree.find_node_by_split(current_pivot_edge)
    destination_order = (
        destination_subtree.get_current_order()
        if destination_subtree is not None
        else destination_tree.get_current_order()
    )
    position_by_taxon = {taxon: index for index, taxon in enumerate(destination_order)}
    fallback_position = len(position_by_taxon)

    order_key: Dict[Partition, Tuple[int, ...]] = {}
    for mover in mover_partitions:
        positions = sorted(
            position_by_taxon[taxon]
            for taxon in mover.taxa
            if taxon in position_by_taxon
        )
        if not positions:
            order_key[mover] = (
                fallback_position,
                fallback_position,
                fallback_position,
                0,
            )
            continue

        order_key[mover] = (
            positions[0],
            positions[-1],
            sum(positions),
            len(positions),
        )

    return order_key


def sort_mover_partitions(
    mover_partitions: set[Partition],
    subtree_order_key: Dict[Partition, Tuple[int, ...]],
) -> list[Partition]:
    return sorted(
        mover_partitions,
        key=lambda partition: (
            *subtree_order_key.get(partition, (MISSING_VISUAL_ORDER,)),
            partition.bitmask,
        ),
    )
