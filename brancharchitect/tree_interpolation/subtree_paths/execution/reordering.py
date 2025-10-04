"""
Partial ordering strategy for subtree interpolation.

This module provides functions to reorder trees during interpolation by focusing
on local subtree contexts to minimize visual disruption.
"""

from __future__ import annotations
import logging
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def reorder_tree_toward_destination(
    source_tree: Node,
    destination_tree: Node,
    active_changing_edge: Partition,
    moving_subtree_partition: Partition,
) -> Node:
    """
    Reorders a subtree by moving a specific 'moving_subtree' block to its
    correct position relative to stable 'anchor' taxa.
    """
    source_subtree = source_tree.find_node_by_split(active_changing_edge)
    dest_subtree = destination_tree.find_node_by_split(active_changing_edge)

    if source_subtree is None or dest_subtree is None:
        logger.warning(
            "Active split not found in one of the trees; skipping reordering."
        )
        return source_tree.deep_copy()

    source_order = list(source_subtree.get_current_order())
    destination_order = list(dest_subtree.get_current_order())
    mover_leaves = set(moving_subtree_partition.taxa)

    # If mover leaves aren't in the source order, something is wrong.
    if not mover_leaves.issubset(set(source_order)):
        logger.warning("Mover leaves not in source order; skipping reordering.")
        return source_tree.deep_copy()

    # 1. Isolate anchors in their original relative order.
    source_anchors = [taxon for taxon in source_order if taxon not in mover_leaves]

    # 2. Find the target insertion index for the mover block within the anchors.
    num_anchors_before = 0
    for taxon in destination_order:
        if taxon in mover_leaves:
            # Found the start of the mover block in the destination.
            # The insertion point is the number of anchors we have seen so far.
            break
        if taxon in source_anchors:
            num_anchors_before += 1

    # 3. Construct the new order.
    new_order = source_anchors
    # The mover block should maintain its internal order from the source tree.
    mover_block_ordered = [taxon for taxon in source_order if taxon in mover_leaves]
    new_order[num_anchors_before:num_anchors_before] = mover_block_ordered

    # 4. Apply the new order to a copy of the tree.
    new_tree = source_tree.deep_copy()
    subtree_node_to_reorder = new_tree.find_node_by_split(active_changing_edge)

    if subtree_node_to_reorder:
        try:
            subtree_node_to_reorder.reorder_taxa(new_order)
        except ValueError as e:
            logger.error(f"Failed to reorder with 'Move the Block' strategy: {e}")
            return source_tree.deep_copy()  # Return original on failure
    return new_tree
