"""
Partial ordering strategy for subtree interpolation.

This module provides functions to reorder trees during interpolation by focusing
on local subtree contexts to minimize visual disruption.
"""

from __future__ import annotations
import logging
from typing import List, Optional

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def reorder_tree_toward_destination(
    source_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    moving_subtree_partition: Partition,
    unstable_taxa: Optional[set[str]] = None,
    copy: bool = True,  # whether to copy the tree first
) -> Node:
    """
    Reorders a subtree by moving a specific jumping-taxa block to its
    correct position relative to stable anchor taxa.

    Args:
        source_tree: The source tree to reorder
        destination_tree: The destination tree to match
        current_pivot_edge: The pivot edge partition
        moving_subtree_partition: The partition of the moving subtree
        unstable_taxa: Optional set of taxa that are moving in this or parallel steps.
                       These should NOT be used as anchors.
        copy: If True, copy the tree first. If False, modify in place.
    """
    source_subtree = source_tree.find_node_by_split(current_pivot_edge)
    dest_subtree = destination_tree.find_node_by_split(current_pivot_edge)

    if source_subtree is None or dest_subtree is None:
        logger.warning(
            "Active split not found in one of the trees; skipping reordering."
        )
        return source_tree  # No modification needed, return original

    source_order = list(source_subtree.get_current_order())
    destination_order = list(dest_subtree.get_current_order())
    mover_leaves = set(moving_subtree_partition.taxa)

    # If no movers, keep subtree stable.
    if not mover_leaves:
        return source_tree  # No modification needed, return original

    # Validate leaf-set/encoding compatibility under the active edge
    if set(source_order) != set(destination_order):
        raise ValueError(
            "Encoding mismatch between source and destination under pivot edge: "
            "leaf sets differ"
        )

    # If jumping-taxa leaves aren't in the source order, something is wrong.
    if not mover_leaves.issubset(set(source_order)):
        logger.warning("Jumping taxa leaves not in source order; skipping reordering.")
        return source_tree  # No modification needed, return original

    logger.debug(f"Reordering for mover {mover_leaves}")
    logger.debug(f"Source Order: {source_order}")
    logger.debug(f"Destination Order: {destination_order}")

    # Identify all unstable taxa (current movers + other simultaneous movers)
    all_unstable = mover_leaves
    if unstable_taxa:
        all_unstable = all_unstable | unstable_taxa

    # Other movers = unstable taxa that are NOT the current mover
    # We must preserve their relative positions in source
    other_movers = all_unstable - mover_leaves

    # 1. Isolate anchors (non-moving, stable taxa) from SOURCE and preserve their order
    source_anchors = [taxon for taxon in source_order if taxon not in all_unstable]
    movers_in_source = [taxon for taxon in source_order if taxon in mover_leaves]

    # Quick optimization: if no anchors, just return the destination order for movers
    # (though typically we assume full leaf set match under pivot)
    if not source_anchors:
        # Just match destination order for these leaves
        new_order = [t for t in destination_order if t in mover_leaves]
        # Or just return destination order if sets match?
        # But we must respect the function contract.
        # Fallback to source relative order if no anchors?
        if not new_order:
            new_order = movers_in_source

        # If there are other movers but no anchors, we might need a fallback strategy.
        # For now, simplistic approach: append others after?
        # Ideally this case (no anchors at all) is rare in local pivots unless root.
    else:
        # 2. Determine the "Anchor Rank" for each mover based on DESTINATION
        # Rank k means "after k-th anchor" (0-indexed).
        # Rank 0: before first anchor. Rank len(anchors): after last anchor.

        anchor_set = set(source_anchors)

        # Map other movers to their SOURCE ranks (to preserve stability)
        other_mover_source_ranks = {}
        source_rank = 0
        for taxon in source_order:
            if taxon in anchor_set:
                source_rank += 1
            elif taxon in other_movers:
                other_mover_source_ranks[taxon] = source_rank

        # Buckets to hold movers for each slot.
        # buckets[i] holds movers that go immediately BEFORE anchor i.
        # buckets[len(source_anchors)] holds movers that go AFTER the last anchor.
        buckets: List[List[str]] = [[] for _ in range(len(source_anchors) + 1)]

        # Destination scan map for CURRENT MOVER only:
        # map each taxon in destination to "number of source-anchors seen so far"
        dest_anchor_rank_map = {}
        current_rank = 0
        for taxon in destination_order:
            if taxon in anchor_set:
                current_rank += 1
            elif taxon in mover_leaves:
                dest_anchor_rank_map[taxon] = current_rank

        # Fill buckets with CURRENT MOVER based on DESTINATION rank
        # We use destination order to ensure that movers in the same bucket
        # are reordered relative to each other to match the destination.
        movers_in_dest = [taxon for taxon in destination_order if taxon in mover_leaves]
        for mover in movers_in_dest:
            # If mover not in dest_anchor_rank_map, it's missing from dest?
            # (Validation above checked set equality, so this shouldn't happen)
            rank = dest_anchor_rank_map.get(mover, 0)
            buckets[rank].append(mover)
            logger.debug(f"Mover {mover} assigned to rank {rank}")

        # Fill buckets with OTHER MOVERS based on SOURCE rank (Preserve Stability)
        # We iterate source order to maintain relative order among other movers too
        for taxon in source_order:
            if taxon in other_movers:
                rank = other_mover_source_ranks.get(taxon, 0)
                buckets[rank].append(taxon)

        # 3. Reconstruct the new order
        new_order = []
        for i in range(len(source_anchors)):
            # Append movers that belong before anchor i
            new_order.extend(buckets[i])
            # Append anchor i
            new_order.append(source_anchors[i])

        # Append remaining movers (after last anchor)
        new_order.extend(buckets[len(source_anchors)])

    # If reordering does nothing, keep original tree
    if new_order == source_order:
        logger.debug("New order identical to source order -> No change.")
        return source_tree  # No change needed, return original

    logger.debug(f"Applying new order: {new_order}")

    # 4. Apply the new order to the tree (copy if requested).
    new_tree = source_tree.deep_copy() if copy else source_tree
    subtree_node_to_reorder = new_tree.find_node_by_split(current_pivot_edge)

    if subtree_node_to_reorder:
        try:
            # Apply the reordering to the entire subtree
            # This uses recursive reorder_taxa to properly order the subtree structure
            subtree_node_to_reorder.reorder_taxa(new_order)
        except ValueError as e:
            logger.error(f"Failed to reorder with 'Move the Block' strategy: {e}")
            return source_tree  # Return original on failure
    return new_tree


def align_to_source_order(
    tree: Node,
    source_order: List[str],
    moving_taxa: Optional[set[str]] = None,
) -> None:
    """
    Align a tree's ordering to match source_order, prioritizing non-moving taxa.

    This function reorders children at each internal node to best match the
    source_order. Unlike reorder_taxa with MINIMUM strategy, it uses a weighted
    approach that strongly prioritizes preserving non-moving taxa positions.

    Args:
        tree: The tree to reorder (modified in place)
        source_order: The target taxa order to match
        moving_taxa: Optional set of taxa that are moving. If provided,
                     non-moving taxa positions are weighted higher.
    """
    if moving_taxa is None:
        moving_taxa = set()

    # Build index map: taxon -> position in source_order
    order_index = {name: i for i, name in enumerate(source_order)}
    n = len(source_order)

    def get_node_sort_key(node: Node) -> tuple[float, int]:
        """
        Compute a sort key for a node based on its leaves' positions in source_order.

        Strategy: Use weighted average of leaf positions, with non-moving taxa
        weighted much higher to preserve their positions.

        Returns a tuple (weighted_avg, min_non_mover_idx) for tie-breaking:
        - Primary: weighted average of positions
        - Secondary: minimum index among non-moving taxa (or first leaf if all movers)
        """
        leaves = node.get_leaves()
        if not leaves:
            return (float("inf"), n)

        total_weight = 0.0
        weighted_sum = 0.0
        min_non_mover_idx = n  # Track minimum index for tie-breaking

        for leaf in leaves:
            idx = order_index.get(leaf.name, n)
            if leaf.name in moving_taxa:
                # Moving taxa get low weight - they should adapt
                weight = 1.0
            else:
                # Non-moving taxa get high weight - they should stay put
                weight = 100.0
                min_non_mover_idx = min(min_non_mover_idx, idx)

            weighted_sum += idx * weight
            total_weight += weight

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else float("inf")

        # If no non-movers, use first leaf index as tie-breaker
        if min_non_mover_idx == n and leaves:
            min_non_mover_idx = order_index.get(leaves[0].name, n)

        return (weighted_avg, min_non_mover_idx)

    def reorder_node(node: Node) -> bool:
        """Recursively reorder children. Returns True if any change occurred."""
        if not node.children:
            return False

        changed = False
        for child in node.children:
            changed = reorder_node(child) or changed

        # Sort children by weighted position
        sorted_children = sorted(node.children, key=get_node_sort_key)

        if sorted_children != node.children:
            node.children = sorted_children
            changed = True

        return changed

    if reorder_node(tree):
        tree.invalidate_caches(propagate_up=True)
