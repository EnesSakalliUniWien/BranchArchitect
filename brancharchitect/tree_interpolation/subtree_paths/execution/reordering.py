"""
Partial ordering strategy for subtree interpolation.

This module provides functions to reorder trees during interpolation by focusing
on local subtree contexts to minimize visual disruption.
"""

from __future__ import annotations
import logging
from typing import List

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def reorder_tree_toward_destination(
    source_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    moving_subtree_partition: Partition,
) -> Node:
    """
    Reorders a subtree by moving a specific jumping-taxa block to its
    correct position relative to stable anchor taxa.
    """
    source_subtree = source_tree.find_node_by_split(current_pivot_edge)
    dest_subtree = destination_tree.find_node_by_split(current_pivot_edge)

    if source_subtree is None or dest_subtree is None:
        logger.warning(
            "Active split not found in one of the trees; skipping reordering."
        )
        return source_tree.deep_copy()

    source_order = list(source_subtree.get_current_order())
    destination_order = list(dest_subtree.get_current_order())
    mover_leaves = set(moving_subtree_partition.taxa)

    # Validate leaf-set/encoding compatibility under the active edge
    if set(source_order) != set(destination_order):
        raise ValueError(
            "Encoding mismatch between source and destination under pivot edge: "
            "leaf sets differ"
        )

    # If jumping-taxa leaves aren't in the source order, something is wrong.
    if not mover_leaves.issubset(set(source_order)):
        logger.warning("Jumping taxa leaves not in source order; skipping reordering.")
        return source_tree.deep_copy()

    # 1. Isolate anchors (non-moving taxa) from SOURCE and preserve their order
    source_anchors = [taxon for taxon in source_order if taxon not in mover_leaves]

    # 2. Find how many anchors precede the mover block in the DESTINATION ordering
    anchors_before_dest: list[str] = []
    for taxon in destination_order:
        if taxon in mover_leaves:
            break
        anchors_before_dest.append(taxon)

    # Determine insertion index by translating destination anchor order back to source order
    anchor_rank_src = {a: i for i, a in enumerate(source_anchors)}
    anchors_before_in_src_order = sorted(
        (a for a in anchors_before_dest if a in anchor_rank_src),
        key=lambda a: anchor_rank_src[a],
    )
    insertion_index = len(anchors_before_in_src_order)

    # 3. Build the new order: anchors remain in SOURCE order, movers inserted en bloc
    new_order: List[str] = list(source_anchors)
    mover_block_ordered = [taxon for taxon in destination_order if taxon in mover_leaves]

    if insertion_index > len(new_order):
        insertion_index = len(new_order)

    new_order[insertion_index:insertion_index] = mover_block_ordered

    # 4. Apply the new order to a copy of the tree.
    new_tree = source_tree.deep_copy()
    subtree_node_to_reorder = new_tree.find_node_by_split(current_pivot_edge)

    if subtree_node_to_reorder:
        try:
            # Apply the reordering to the entire subtree
            # This uses recursive reorder_taxa to properly order the subtree structure
            subtree_node_to_reorder.reorder_taxa(new_order)
        except ValueError as e:
            logger.error(f"Failed to reorder with 'Move the Block' strategy: {e}")
            return source_tree.deep_copy()  # Return original on failure
    return new_tree
