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
    unstable_mover_partitions: Optional[List[Partition]] = None,
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    copy: bool = True,  # whether to copy the tree first
) -> Node:
    """
    Reorder one interpolation microstep.

    Contract:
    - moving_subtree_partition is the only block that moves in this call.
    - unstable_mover_partitions is context for anchor selection only: taxa in
      those partitions are not stable anchors, but they do not move unless they
      are also moving_subtree_partition.

    Block-aware algorithm:
    1. Treats the current mover as a cohesive block.
    2. Uses taxa outside all unstable movers as anchors.
    3. Places the current mover at its destination position.
    4. Preserves inactive mover taxa at their current/source anchor ranks.
    5. Preserves source order within the current mover block.

    Args:
        source_tree: The source tree to reorder
        destination_tree: The destination tree to match
        current_pivot_edge: The pivot edge partition
        moving_subtree_partition: The only subtree partition that moves now.
        unstable_mover_partitions: All mover partitions for this pivot edge.
                                   Used only to exclude unstable taxa from anchors.
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

    current_order = list(source_subtree.get_current_order())
    target_order = list(dest_subtree.get_current_order())

    active_mover_taxa = set(moving_subtree_partition.taxa)

    # If no movers, keep subtree stable.
    if not active_mover_taxa:
        return source_tree  # No modification needed, return original

    # Validate leaf-set/encoding compatibility under the active edge
    if set(current_order) != set(target_order):
        raise ValueError(
            "Encoding mismatch between source and destination under pivot edge: "
            "leaf sets differ"
        )

    # If jumping-taxa leaves aren't in the source order, something is wrong.
    if not active_mover_taxa.issubset(set(current_order)):
        logger.warning("Jumping taxa leaves not in source order; skipping reordering.")
        return source_tree  # No modification needed, return original

    logger.debug(f"Reordering active mover block {active_mover_taxa}")
    logger.debug(f"Current order: {current_order}")
    logger.debug(f"Target order: {target_order}")

    mover_context = list(unstable_mover_partitions or [moving_subtree_partition])
    if all(p.bitmask != moving_subtree_partition.bitmask for p in mover_context):
        mover_context.append(moving_subtree_partition)

    unstable_mover_taxa: set[str] = set()
    for partition in mover_context:
        unstable_mover_taxa.update(partition.taxa)
    inactive_mover_taxa = unstable_mover_taxa - active_mover_taxa

    # Stable anchors are taxa that are not part of any mover for this pivot edge.
    anchor_taxa = [taxon for taxon in current_order if taxon not in unstable_mover_taxa]
    active_mover_order = [
        taxon for taxon in current_order if taxon in active_mover_taxa
    ]
    target_position = {taxon: index for index, taxon in enumerate(target_order)}

    # No anchors: keep inactive movers at their current rank and insert only the
    # active mover block relative to them using target order.
    if not anchor_taxa:
        new_order = [taxon for taxon in current_order if taxon in inactive_mover_taxa]
        insert_at = _target_bucket_insert_index(
            bucket_taxa=new_order,
            active_mover_order=active_mover_order,
            target_position=target_position,
        )
        new_order[insert_at:insert_at] = active_mover_order
    else:
        # 2. Block-aware bucketing
        #
        # Key insight: only the active mover gets a destination anchor rank.
        # Inactive movers stay at their current anchor ranks so later microsteps
        # can move them explicitly.

        # Buckets: buckets[i] holds taxa that go immediately BEFORE anchor i
        # buckets[len(anchors)] holds taxa that go AFTER the last anchor
        buckets: List[List[str]] = [[] for _ in range(len(anchor_taxa) + 1)]

        anchor_set = set(anchor_taxa)
        inactive_anchor_rank_by_taxon: dict[str, int] = {}
        current_anchor_rank = 0
        for taxon in current_order:
            if taxon in anchor_set:
                current_anchor_rank += 1
            elif taxon in inactive_mover_taxa:
                inactive_anchor_rank_by_taxon[taxon] = current_anchor_rank

        active_destination_rank = _compute_destination_rank_from_order(
            destination_order=target_order,
            anchor_taxa=anchor_taxa,
            current_mover_taxa=active_mover_taxa,
        )

        for taxon in current_order:
            if taxon in inactive_mover_taxa:
                buckets[inactive_anchor_rank_by_taxon.get(taxon, 0)].append(taxon)

        # If inactive movers already occupy the target bucket, insert the active
        # mover on the target-order side of those inactive movers. The inactive
        # movers keep their current anchor rank; only the active block is placed.
        active_bucket = buckets[active_destination_rank]
        insert_at = _target_bucket_insert_index(
            bucket_taxa=active_bucket,
            active_mover_order=active_mover_order,
            target_position=target_position,
        )
        active_bucket[insert_at:insert_at] = active_mover_order

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
    if new_order == current_order:
        logger.debug("New order identical to source order -> No change.")
        return source_tree  # No change needed, return original

    # 4. Apply the new order to the tree (copy if requested).
    new_tree = source_tree.deep_copy() if copy else source_tree
    subtree_node_to_reorder = new_tree.find_node_by_split(current_pivot_edge)

    if subtree_node_to_reorder:
        try:
            # Apply the reordering to the entire subtree
            # This uses recursive reorder_taxa to properly order the subtree structure
            subtree_node_to_reorder.reorder_taxa(new_order)
        except ValueError as e:
            raise ValueError("Failed to reorder with 'Move the Block' strategy") from e
    return new_tree


def _target_bucket_insert_index(
    bucket_taxa: List[str],
    active_mover_order: List[str],
    target_position: Dict[str, int],
) -> int:
    """Return where the active mover block belongs inside a shared anchor bucket."""
    if not active_mover_order:
        return len(bucket_taxa)

    active_target_position = min(target_position[taxon] for taxon in active_mover_order)
    return sum(
        1
        for taxon in bucket_taxa
        if target_position.get(taxon, len(target_position)) < active_target_position
    )


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
