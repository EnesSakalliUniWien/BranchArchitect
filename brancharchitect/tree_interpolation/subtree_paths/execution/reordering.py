"""
Partial ordering strategy for subtree interpolation.

This module provides functions to reorder trees during interpolation by focusing
on local subtree contexts to minimize visual disruption.

Key Algorithm: Anchor-Based Bucket Placement
---------------------------------------------
When moving a subtree (the "mover") to its destination position:
1. Identify "anchors" - stable taxa that define the skeleton structure
2. Compute anchor ranks in the destination (which anchor slot each mover goes to)
3. Place movers into buckets based on their destination anchor rank
4. Reconstruct the order: [bucket_0, anchor_0, bucket_1, anchor_1, ..., bucket_n]

This preserves anchor ordering while moving the mover subtree to its correct slot.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

logger = logging.getLogger(__name__)


# =============================================================================
# Main Reordering Function
# =============================================================================


def reorder_tree_toward_destination(
    source_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    moving_subtree_partition: Partition,
    unstable_taxa: Optional[Set[str]] = None,
    copy: bool = True,
) -> Node:
    """
    Reorder a subtree by placing a moving block at its destination position.

    This function uses anchor-based bucket placement:
    - Anchors (stable taxa) define the skeleton in destination order
    - The moving subtree is placed into the correct bucket (slot between anchors)
    - Other unstable taxa preserve their source positions

    Args:
        source_tree: The tree to reorder.
        destination_tree: The target tree defining destination positions.
        current_pivot_edge: The pivot edge partition (defines the subtree scope).
        moving_subtree_partition: The partition of the subtree being moved.
        unstable_taxa: Taxa still moving in parallel steps (not used as anchors).
        copy: If True, return a copy; if False, modify in place.

    Returns:
        The reordered tree (copy or modified original).

    Raises:
        ValueError: If source and destination have different leaf sets under pivot.
    """
    # --- Step 0: Locate subtrees under pivot edge ---
    source_subtree = source_tree.find_node_by_split(current_pivot_edge)
    dest_subtree = destination_tree.find_node_by_split(current_pivot_edge)

    if source_subtree is None or dest_subtree is None:
        logger.warning("Pivot edge not found in one of the trees; skipping reordering.")
        return source_tree

    # --- Step 1: Extract current and target leaf orders ---
    source_order = list(source_subtree.get_current_order())
    destination_order = list(dest_subtree.get_current_order())
    mover_taxa = set(moving_subtree_partition.taxa)

    # Early exit: nothing to move
    if not mover_taxa:
        return source_tree

    # Validate leaf sets match
    if set(source_order) != set(destination_order):
        raise ValueError(
            "Leaf set mismatch between source and destination under pivot edge"
        )

    # Validate movers exist in source
    if not mover_taxa.issubset(set(source_order)):
        logger.warning("Moving taxa not found in source order; skipping reordering.")
        return source_tree

    logger.info(f"Reordering for mover {mover_taxa}")
    logger.info(f"Source Order: {source_order}")
    logger.info(f"Destination Order: {destination_order}")

    # --- Step 2: Classify taxa ---
    all_unstable_taxa = mover_taxa | (unstable_taxa or set())
    other_unstable_taxa = all_unstable_taxa - mover_taxa

    # Anchors: stable taxa in DESTINATION order (defines the skeleton)
    anchors = [t for t in destination_order if t not in all_unstable_taxa]
    anchor_set = set(anchors)

    # Movers in their source order (preserves internal ordering)
    movers_in_source_order = [t for t in source_order if t in mover_taxa]

    # --- Step 3: Compute new order using bucket placement ---
    if not anchors:
        # Edge case: no anchors means all taxa are unstable
        # Use destination order for current movers
        new_order = [t for t in destination_order if t in mover_taxa]
        if not new_order:
            new_order = movers_in_source_order
    else:
        new_order = _compute_bucket_order(
            anchors=anchors,
            anchor_set=anchor_set,
            movers=movers_in_source_order,
            mover_taxa=mover_taxa,
            other_unstable=other_unstable_taxa,
            source_order=source_order,
            destination_order=destination_order,
        )

    # --- Step 4: Apply if changed ---
    if new_order == source_order:
        logger.info("New order identical to source order -> No change.")
        return source_tree

    logger.info(f"Applying new order: {new_order}")

    new_tree = source_tree.deep_copy() if copy else source_tree
    subtree_to_reorder = new_tree.find_node_by_split(current_pivot_edge)

    if subtree_to_reorder:
        try:
            subtree_to_reorder.reorder_taxa(new_order)
        except ValueError as e:
            logger.error(f"Failed to apply reordering: {e}")
            return source_tree

    return new_tree


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_bucket_order(
    anchors: List[str],
    anchor_set: Set[str],
    movers: List[str],
    mover_taxa: Set[str],
    other_unstable: Set[str],
    source_order: List[str],
    destination_order: List[str],
) -> List[str]:
    """
    Compute the new leaf order using anchor-based bucket placement.

    The algorithm:
    1. Create buckets[0..n] where n = len(anchors)
       - bucket[i] holds taxa that appear BEFORE anchor[i] in destination
       - bucket[n] holds taxa that appear AFTER the last anchor
    2. Place current movers based on their DESTINATION rank
    3. Place other unstable taxa based on their SOURCE rank (preserve stability)
    4. Reconstruct: [bucket_0, anchor_0, bucket_1, anchor_1, ..., bucket_n]

    Args:
        anchors: Stable taxa in destination order (the skeleton).
        anchor_set: Set of anchor taxa for O(1) lookup.
        movers: Current moving taxa in source order.
        mover_taxa: Set of current moving taxa.
        other_unstable: Other unstable taxa (not current movers).
        source_order: Current leaf order in source tree.
        destination_order: Target leaf order in destination tree.

    Returns:
        The computed new leaf order.
    """
    num_buckets = len(anchors) + 1
    buckets: List[List[str]] = [[] for _ in range(num_buckets)]

    # --- Compute destination ranks for current movers ---
    # Rank = number of anchors seen before this taxon in destination
    dest_rank_for_movers = _compute_anchor_ranks(
        order=destination_order,
        anchor_set=anchor_set,
        target_taxa=mover_taxa,
    )

    # --- Compute source ranks for other unstable taxa ---
    # These preserve their source positions relative to anchors
    source_rank_for_others = _compute_anchor_ranks(
        order=source_order,
        anchor_set=anchor_set,
        target_taxa=other_unstable,
    )

    # --- Fill buckets ---
    # Current movers go to their DESTINATION rank
    for taxon in movers:
        rank = dest_rank_for_movers.get(taxon, 0)
        buckets[rank].append(taxon)
        logger.debug(f"Mover '{taxon}' -> bucket {rank}")

    # Other unstable taxa go to their SOURCE rank (preserve stability)
    for taxon in source_order:
        if taxon in other_unstable:
            rank = source_rank_for_others.get(taxon, 0)
            buckets[rank].append(taxon)

    # --- Reconstruct order ---
    new_order: List[str] = []
    for i, anchor in enumerate(anchors):
        new_order.extend(buckets[i])  # Taxa before this anchor
        new_order.append(anchor)  # The anchor itself
    new_order.extend(buckets[len(anchors)])  # Taxa after last anchor

    return new_order


def _compute_anchor_ranks(
    order: List[str],
    anchor_set: Set[str],
    target_taxa: Set[str],
) -> dict[str, int]:
    """
    Compute the anchor rank for each target taxon in a given order.

    Anchor rank = number of anchors seen before the taxon.
    - Rank 0: before first anchor
    - Rank k: after k-th anchor (0-indexed: after anchors[k-1])
    - Rank n: after last anchor

    Args:
        order: The leaf order to scan.
        anchor_set: Set of anchor taxa.
        target_taxa: Taxa to compute ranks for.

    Returns:
        Dict mapping taxon -> anchor rank.
    """
    rank_map: dict[str, int] = {}
    current_rank = 0

    for taxon in order:
        if taxon in anchor_set:
            current_rank += 1
        elif taxon in target_taxa:
            rank_map[taxon] = current_rank

    return rank_map


# =============================================================================
# Alternative Alignment Strategy
# =============================================================================


def align_to_source_order(
    tree: Node,
    source_order: List[str],
    moving_taxa: Optional[Set[str]] = None,
) -> None:
    """
    Align a tree's ordering to match source_order using weighted positioning.

    This function reorders children at each internal node to best match the
    source_order. Non-moving taxa are weighted heavily to preserve their
    positions, while moving taxa adapt around them.

    Args:
        tree: The tree to reorder (modified in place).
        source_order: The target taxa order to match.
        moving_taxa: Taxa that are moving (weighted lower).
    """
    if moving_taxa is None:
        moving_taxa = set()

    order_index = {name: i for i, name in enumerate(source_order)}
    n = len(source_order)

    # Weight constants
    MOVER_WEIGHT = 1.0
    STABLE_WEIGHT = 100.0

    def get_sort_key(node: Node) -> tuple[float, int]:
        """
        Compute sort key based on weighted average of leaf positions.

        Returns:
            (weighted_average, min_stable_index) for sorting.
        """
        leaves = node.get_leaves()
        if not leaves:
            return (float("inf"), n)

        total_weight = 0.0
        weighted_sum = 0.0
        min_stable_idx = n

        for leaf in leaves:
            idx = order_index.get(leaf.name, n)
            if leaf.name in moving_taxa:
                weight = MOVER_WEIGHT
            else:
                weight = STABLE_WEIGHT
                min_stable_idx = min(min_stable_idx, idx)

            weighted_sum += idx * weight
            total_weight += weight

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else float("inf")

        # Fallback tie-breaker if no stable taxa
        if min_stable_idx == n and leaves:
            min_stable_idx = order_index.get(leaves[0].name, n)

        return (weighted_avg, min_stable_idx)

    def reorder_recursively(node: Node) -> bool:
        """Recursively reorder children. Returns True if changed."""
        if not node.children:
            return False

        changed = any(reorder_recursively(child) for child in node.children)

        sorted_children = sorted(node.children, key=get_sort_key)
        if sorted_children != node.children:
            node.children = sorted_children
            changed = True

        return changed

    if reorder_recursively(tree):
        tree.invalidate_caches(propagate_up=True)
