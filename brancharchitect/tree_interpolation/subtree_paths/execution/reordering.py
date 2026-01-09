"""
Partial ordering strategy for subtree interpolation.

This module provides functions to reorder trees during interpolation by focusing
on local subtree contexts to minimize visual disruption.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def reorder_tree_toward_destination(
    source_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    moving_subtree_partition: Partition,
    all_mover_partitions: Optional[List[Partition]] = None,
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    copy: bool = True,  # whether to copy the tree first
) -> Node:
    """
    Reorders a subtree by moving a specific jumping-taxa block to its
    correct position relative to stable anchor taxa.

    MRCA-aware algorithm (when parent maps provided):
    1. Uses destination parent position to determine block placement
    2. Falls back to first-occurrence method if no parent maps

    Block-aware algorithm:
    1. Treats each mover partition as a cohesive BLOCK (not individual taxa)
    2. Uses taxa NOT in any mover block as anchors
    3. Places the moving block at its destination position
    4. Preserves other mover blocks at their SOURCE positions (stability)
    5. Preserves SOURCE order within the moving block

    Args:
        source_tree: The source tree to reorder
        destination_tree: The destination tree to match
        current_pivot_edge: The pivot edge partition
        moving_subtree_partition: The partition of the moving subtree (block)
        all_mover_partitions: List of all moving subtree Partitions (blocks).
                              Used to identify stable anchor taxa.
        source_parent_map: Maps each mover -> its parent in source tree (MRCA).
        dest_parent_map: Maps each mover -> its parent in destination tree (MRCA).
        copy: If True, copy the tree first. If False, modify in place.
    """
    source_subtree = source_tree.find_node_by_split(current_pivot_edge)
    dest_subtree = destination_tree.find_node_by_split(current_pivot_edge)

    if source_parent_map or dest_parent_map:
        logger.debug(
            f"Reordering with provided parent maps. "
            f"Source: {source_parent_map.get(moving_subtree_partition) if source_parent_map else 'N/A'}, "
            f"Dest: {dest_parent_map.get(moving_subtree_partition) if dest_parent_map else 'N/A'}"
        )

    if source_subtree is None or dest_subtree is None:
        logger.warning(
            "Active split not found in one of the trees; skipping reordering."
        )
        return source_tree  # No modification needed, return original

    source_order = list(source_subtree.get_current_order())
    destination_order = list(dest_subtree.get_current_order())

    # Current mover block's taxa
    current_mover_taxa = set(moving_subtree_partition.taxa)

    # If no movers, keep subtree stable.
    if not current_mover_taxa:
        return source_tree  # No modification needed, return original

    # Validate leaf-set/encoding compatibility under the active edge
    if set(source_order) != set(destination_order):
        raise ValueError(
            "Encoding mismatch between source and destination under pivot edge: "
            "leaf sets differ"
        )

    # If jumping-taxa leaves aren't in the source order, something is wrong.
    if not current_mover_taxa.issubset(set(source_order)):
        logger.warning("Jumping taxa leaves not in source order; skipping reordering.")
        return source_tree  # No modification needed, return original

    logger.debug(f"Reordering for mover block {current_mover_taxa}")
    logger.debug(f"Source Order: {source_order}")
    logger.debug(f"Destination Order: {destination_order}")

    # Build the set of ALL unstable taxa (all mover blocks combined)
    all_mover_taxa: set[str] = set()
    if all_mover_partitions:
        for partition in all_mover_partitions:
            all_mover_taxa.update(partition.taxa)
    else:
        # Fallback: only current mover is unstable
        all_mover_taxa = current_mover_taxa

    # Other mover blocks' taxa (not the current mover)
    other_mover_taxa = all_mover_taxa - current_mover_taxa

    # 1. Identify ANCHOR taxa (stable, not in any mover block)
    anchor_taxa = [taxon for taxon in source_order if taxon not in all_mover_taxa]

    # Quick optimization: if no anchors, use destination order for current mover
    if not anchor_taxa:
        # Use destination order for current mover, then other movers in source order
        new_order = [t for t in destination_order if t in current_mover_taxa]
        # Append other movers in their source order
        new_order.extend([t for t in source_order if t in other_mover_taxa])
    else:
        # 2. Block-aware bucketing
        #
        # Key insight: We place the ENTIRE current mover block at the position
        # determined by where its taxa appear in the destination (relative to anchors).
        # Other mover blocks stay at their SOURCE positions (stability).

        anchor_set = set(anchor_taxa)

        # Compute anchor rank for each OTHER mover taxon based on SOURCE
        # (preserves their relative positions)
        other_mover_source_ranks: dict[str, int] = {}
        source_rank = 0
        for taxon in source_order:
            if taxon in anchor_set:
                source_rank += 1
            elif taxon in other_mover_taxa:
                other_mover_source_ranks[taxon] = source_rank

        # Find the BLOCK's destination rank
        # Use destination-order scanning logic (previously V1 fallback) which is robust for
        # both sibling reordering and global placement relative to anchors.
        block_dest_rank = _compute_destination_rank_from_order(
            destination_order=destination_order,
            anchor_taxa=anchor_taxa,
            current_mover_taxa=current_mover_taxa,
        )

        logger.debug(f"Block destination rank: {block_dest_rank}")

        # Buckets: buckets[i] holds taxa that go immediately BEFORE anchor i
        # buckets[len(anchors)] holds taxa that go AFTER the last anchor
        buckets: List[List[str]] = [[] for _ in range(len(anchor_taxa) + 1)]

        # Place ENTIRE current mover block at its destination rank
        # But preserve SOURCE order within the block (internal structure stays same)
        current_mover_in_source = [t for t in source_order if t in current_mover_taxa]
        buckets[block_dest_rank].extend(current_mover_in_source)

        # Place OTHER mover taxa at their SOURCE ranks (stability)
        for taxon in source_order:
            if taxon in other_mover_taxa:
                rank = other_mover_source_ranks.get(taxon, 0)
                buckets[rank].append(taxon)

        # 3. Reconstruct the new order
        new_order = []
        for i in range(len(anchor_taxa)):
            # Append movers that belong before anchor i
            new_order.extend(buckets[i])
            # Append anchor i
            new_order.append(anchor_taxa[i])

        # Append remaining movers (after last anchor)
        new_order.extend(buckets[len(anchor_taxa)])

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


def _compute_destination_rank_from_order(
    destination_order: List[str],
    anchor_taxa: List[str],
    current_mover_taxa: set[str],
) -> int:
    """
    Compute where the mover block should be placed among anchors based on destination order.

    Scans the destination order to find the first occurrence of the mover block
    relative to the anchors.
    """
    anchor_set = set(anchor_taxa)

    dest_anchor_rank_map: dict[str, int] = {}
    current_rank = 0
    for taxon in destination_order:
        if taxon in anchor_set:
            current_rank += 1
        elif taxon in current_mover_taxa:
            dest_anchor_rank_map[taxon] = current_rank

    if dest_anchor_rank_map:
        return min(dest_anchor_rank_map.values())
    return 0


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
