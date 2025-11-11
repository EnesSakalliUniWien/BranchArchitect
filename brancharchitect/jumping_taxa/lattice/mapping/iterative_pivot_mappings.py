"""Iterative pivot-edge mapping for the lattice algorithm.

This module implements the mapping of pivot edges found in pruned trees back to
their corresponding splits in the original unpruned trees using subset containment.

Mathematical Framework
----------------------
Let T₁, T₂ be the original input trees with leaf set L.
Let T₁⁽ⁱ⁾, T₂⁽ⁱ⁾ be the trees at iteration i with leaf set Lⁱ ⊆ L.
Let Σ(T) denote the set of all splits in tree T.

For a pivot edge p ∈ Σ(T₁⁽ⁱ⁾) ∩ Σ(T₂⁽ⁱ⁾), we seek a split s ∈ Σ(T₁) ∩ Σ(T₂)
such that p ⊆ s when both are viewed as subsets of L.

The mapping function Φ: Σ(T₁⁽ⁱ⁾) ∩ Σ(T₂⁽ⁱ⁾) → Σ(T₁) ∩ Σ(T₂) is defined as:

For direct pivot edges (no jumping taxa, J = ∅):
    Φ(p) = argmax_{s ∈ Σ(T₁) ∩ Σ(T₂), p ⊆ s} |s|
    (Maps to the MAXIMUM/largest containing split)

For pivot edges with jumping taxa (J ≠ ∅):
    Φ(p, J) = argmin_{s ∈ Σ(T₁) ∩ Σ(T₂), (p ∪ J) ⊆ s} |s|
    (Maps to the MINIMUM/smallest containing split)

where |s| denotes the cardinality of the split (number of taxa).

Rationale
---------
- Direct pivot edges map to maximum splits to allow broader, higher-level
  structural changes in the tree interpolation
- Pivot edges with jumping taxa map to minimum splits to find the most
  specific split that contains both the pivot and the jumping taxa

Implementation Notes
--------------------
- Splits are represented as bitmasks over the original leaf indexing
- Subset relation p ⊆ s is checked via bitwise AND: (p & s) == p
- Among splits of equal size, ties are broken deterministically by bitmask value
"""

from typing import List, Optional, Set

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition, PartitionSet
from brancharchitect.jumping_taxa.debug import jt_logger


def _popcount(bitmask: int) -> int:
    """
    Count the number of set bits in a bitmask.

    This gives the cardinality |p| of a partition p represented as a bitmask.

    Args:
        bitmask: Integer bitmask representation of a partition

    Returns:
        Number of 1-bits in the bitmask
    """
    try:
        return bitmask.bit_count()  # Python 3.10+
    except AttributeError:
        return bin(bitmask).count("1")  # Fallback for older Python


def map_iterative_pivot_edges_to_original(
    pivot_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
    current_t1: Node | None = None,
    current_t2: Node | None = None,
    jumping_taxa_solutions: List[List[Partition]] | None = None,
) -> List[Partition]:
    """
    Map pivot edges from pruned trees to their corresponding nodes in original trees.

    This function implements a dual mapping strategy based on whether the pivot edge
    has associated jumping taxa:

    For direct pivot edges (no jumping taxa):
        Find the MAXIMUM (largest) common split s in the original trees such that p ⊆ s.
        This allows direct pivots to map to broader, higher-level splits.

    For pivot edges with jumping taxa:
        Find the MINIMUM (smallest) common split s in the original trees such that
        (p ∪ J) ⊆ s, where J is the set of jumping taxa.
        This finds the most specific split containing both pivot and jumping taxa.

    Algorithm:
        For each pivot edge p with optional jumping taxa solution J:
        1. Calculate the expected original taxa set: E = p ∪ J.
        2. Find all common splits C in the original trees (T₁ ∩ T₂).
        3. Find all candidate splits s ∈ C such that E is a subset of s.
        4. If J = ∅ (direct pivot): select the split with MAXIMUM cardinality.
           If J ≠ ∅ (jumping taxa): select the split with MINIMUM cardinality.

    Args:
        pivot_edges_from_iteration: Pivot edges p from the current iteration.
        original_t1: Original unpruned tree T₁.
        original_t2: Original unpruned tree T₂.
        current_t1: Current pruned tree T₁⁽ⁱ⁾ (unused, for API compatibility).
        current_t2: Current pruned tree T₂⁽ⁱ⁾ (unused, for API compatibility).
        jumping_taxa_solutions: Flat list of partitions per pivot edge, if available.

    Returns:
        List of mapped splits. Every pivot edge is guaranteed to be mapped to a valid split,
        using the root split as a fallback if necessary.
    """
    mapped_splits: List[Partition] = []

    # Pre-compute all common splits from the original trees for efficient lookup.
    # This is the set C = Σ(T₁) ∩ Σ(T₂).
    original_common_splits: PartitionSet[Partition] = (
        original_t1.to_splits() & original_t2.to_splits()
    )

    # Determine the total number of taxa to identify the root split
    # The root split contains all taxa in the tree
    total_taxa = len(original_t1.taxa_encoding)

    # Separate root split from other common splits
    # Pivot edges should map to internal splits, not the trivial root split
    root_split: Optional[Partition] = None
    non_root_common_splits: List[Partition] = []

    for split in original_common_splits:
        if len(split.indices) == total_taxa:
            root_split = split
        else:
            non_root_common_splits.append(split)

    jt_logger.info(f"\n{'=' * 80}")
    jt_logger.info(f"Number of pivot edges to map: {len(pivot_edges_from_iteration)}")
    jt_logger.info(f"Total taxa in original trees: {total_taxa}")
    jt_logger.info(
        f"Number of common splits in original trees: {len(original_common_splits)}"
    )
    jt_logger.info(f"Number of non-root common splits: {len(non_root_common_splits)}")
    if root_split:
        jt_logger.info(f"Root split (all {total_taxa} taxa) available as fallback")
    jt_logger.info(f"{'=' * 80}\n")

    for i, pivot_edge in enumerate(pivot_edges_from_iteration):
        jt_logger.info(f"\n[MAPPING DEBUG] Processing pivot edge {i}")
        jt_logger.info(f"[MAPPING DEBUG]   Pivot edge: {pivot_edge}")
        jt_logger.info(f"[MAPPING DEBUG]   Pivot indices: {sorted(pivot_edge.indices)}")
        jt_logger.info(f"[MAPPING DEBUG]   Pivot size: {len(pivot_edge.indices)} taxa")

        # Get the jumping taxa solution J for this pivot edge as a flat list of partitions.
        jumping_taxa_indices: Set[int] = set()
        if jumping_taxa_solutions and i < len(jumping_taxa_solutions):
            partitions_for_this_pivot = jumping_taxa_solutions[i]
            jt_logger.info(
                f"[MAPPING DEBUG]   Number of solution partitions for this pivot: {len(partitions_for_this_pivot)}"
            )
            if partitions_for_this_pivot:
                for j, partition in enumerate(partitions_for_this_pivot):
                    jt_logger.info(
                        f"[MAPPING DEBUG]     Solution partition {j}: {sorted(partition.indices)}"
                    )
                    jumping_taxa_indices.update(partition.indices)

        jt_logger.info(
            f"[MAPPING DEBUG]   Jumping taxa indices: {sorted(jumping_taxa_indices)}"
        )
        jt_logger.info(
            f"[MAPPING DEBUG]   Jumping taxa count: {len(jumping_taxa_indices)}"
        )

        # Step 1: Calculate the expected original taxa set (E = p ∪ J).
        pivot_indices: Set[int] = set(pivot_edge.indices)
        expected_original_indices: Set[int] = pivot_indices | jumping_taxa_indices

        jt_logger.info(
            f"[MAPPING DEBUG]   Expected original indices (pivot ∪ jumping): {sorted(expected_original_indices)}"
        )
        jt_logger.info(
            f"[MAPPING DEBUG]   Expected original size: {len(expected_original_indices)} taxa"
        )

        # Create a bitmask for the expected taxa set for efficient subset checking.
        expected_bitmask: int = 0
        for index in expected_original_indices:
            expected_bitmask |= 1 << index

        jt_logger.info(
            f"[MAPPING DEBUG]   Expected bitmask popcount: {_popcount(expected_bitmask)}"
        )

        # Step 2: Find all common splits that are supersets of the expected taxa set.
        # IMPORTANT: First try to find containing splits from non-root splits only.
        # Pivot edges should map to meaningful internal splits, not the trivial root.
        containing_splits: List[Partition] = [
            split
            for split in non_root_common_splits
            if (expected_bitmask & split.bitmask) == expected_bitmask
        ]

        jt_logger.info(
            f"[MAPPING DEBUG]   Found {len(containing_splits)} non-root containing splits"
        )

        # If no non-root splits found, check if root split can contain it (fallback)
        if not containing_splits and root_split:
            if (expected_bitmask & root_split.bitmask) == expected_bitmask:
                containing_splits = [root_split]
                jt_logger.warning(
                    "⚠️  Only root split contains required taxa - using as fallback"
                )

        if containing_splits:
            jt_logger.info(
                f"[MAPPING DEBUG]   Containing split sizes: {[len(s.indices) for s in containing_splits]}"
            )
            # Show the smallest 3
            smallest = sorted(
                containing_splits, key=lambda s: (_popcount(s.bitmask), s.bitmask)
            )[:3]
            for j, split in enumerate(smallest):
                jt_logger.info(
                    f"[MAPPING DEBUG]     Candidate {j}: size={len(split.indices)}, indices={sorted(split.indices)}"
                )

        # Step 3: From the valid candidates, select the appropriate one.
        # For direct pivot edges (no jumping taxa), use the MAXIMUM (largest) split.
        # For pivot edges with jumping taxa, use the MINIMUM (smallest) split.
        if containing_splits:
            # Check if this is a direct pivot edge (no jumping taxa)
            is_direct_pivot = len(jumping_taxa_indices) == 0

            if is_direct_pivot:
                # Direct pivot edge: map to the LARGEST containing split
                selected_split: Partition = max(
                    containing_splits, key=lambda s: (_popcount(s.bitmask), s.bitmask)
                )
                jt_logger.info("✓ Direct pivot edge - mapping to MAXIMUM split")
            else:
                # Pivot with jumping taxa: map to the SMALLEST containing split
                selected_split: Partition = min(
                    containing_splits, key=lambda s: (_popcount(s.bitmask), s.bitmask)
                )
                jt_logger.info("✓ Pivot with jumping taxa - mapping to MINIMUM split")

            jt_logger.info(f"[MAPPING DEBUG]   ✓ Mapped to split: {selected_split}")
            jt_logger.info(
                f"[MAPPING DEBUG]   ✓ Mapped split indices: {sorted(selected_split.indices)}"
            )
            jt_logger.info(
                f"[MAPPING DEBUG]   ✓ Mapped split size: {len(selected_split.indices)} taxa"
            )
            mapped_splits.append(selected_split)
        else:
            # If no common split contains the expected taxa, fall back to root split.
            # This ensures we always return a valid split for every pivot edge.
            if root_split:
                jt_logger.warning(
                    "[MAPPING DEBUG]   ⚠️  No containing split found - falling back to root split"
                )
                mapped_splits.append(root_split)
            else:
                # This should never happen if the trees are valid
                jt_logger.error(
                    "[MAPPING DEBUG]   ✗ CRITICAL: No root split available! Using first common split as fallback"
                )
                mapped_splits.append(
                    list(original_common_splits)[0]
                    if original_common_splits
                    else pivot_edge
                )

    return mapped_splits
