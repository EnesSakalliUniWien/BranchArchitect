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
    Φ(p) = argmin_{s ∈ Σ(T₁) ∩ Σ(T₂), p ⊆ s} |s|
    (Maps to the MINIMUM/smallest containing split)

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
from brancharchitect.logger.debug import jt_logger


def map_iterative_pivot_edges_to_original(
    pivot_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
    jumping_taxa_solutions: List[List[Partition]] | None = None,
) -> List[Partition]:
    """
    Map pivot edges from pruned trees to their corresponding nodes in original trees.

    This function implements a dual mapping strategy based on whether the pivot edge
    has associated jumping taxa:

    For direct pivot edges (no jumping taxa):
        Find the MINIMUM (smallest) common split s in the original trees such that p ⊆ s.
        This ensures topological precision and deep common ancestor mapping.

    For pivot edges with jumping taxa:
        Find the MINIMUM (smallest) common split s in the original trees such that
        (p ∪ J) ⊆ s, where J is the set of jumping taxa.
        This finds the most specific split containing both pivot and jumping taxa.

    Algorithm:
        For each pivot edge p with optional jumping taxa solution J:
        1. Calculate the expected original taxa set: E = p ∪ J.
        2. Find all common splits C in the original trees (T₁ ∩ T₂).
        3. Find all candidate splits s ∈ C such that E is a subset of s.
        4. Select the split with MINIMUM cardinality.

    Args:
        pivot_edges_from_iteration: Pivot edges p from the current iteration.
        original_t1: Original unpruned tree T₁.
        original_t2: Original unpruned tree T₂.
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

    if not jt_logger.disabled:
        jt_logger.info(
            f"Mapping {len(pivot_edges_from_iteration)} pivot edges "
            f"({len(original_common_splits)} common splits available, "
            f"{len(non_root_common_splits)} non-root)"
        )

    success_count = 0
    fallback_count = 0

    for i, pivot_edge in enumerate(pivot_edges_from_iteration):
        # Get the jumping taxa solution J for this pivot edge as a flat list of partitions.
        jumping_taxa_indices: Set[int] = set()
        if jumping_taxa_solutions and i < len(jumping_taxa_solutions):
            partitions_for_this_pivot = jumping_taxa_solutions[i]
            if partitions_for_this_pivot:
                for partition in partitions_for_this_pivot:
                    jumping_taxa_indices.update(partition.indices)

        # Step 1: Calculate the expected original taxa set (E = p ∪ J).
        pivot_indices: Set[int] = set(pivot_edge.indices)
        expected_original_indices: Set[int] = pivot_indices | jumping_taxa_indices

        # Create a bitmask for the expected taxa set for efficient subset checking.
        expected_bitmask: int = 0
        for index in expected_original_indices:
            expected_bitmask |= 1 << index

        # Step 2: Find all common splits that are supersets of the expected taxa set.
        containing_splits: List[Partition] = [
            split
            for split in non_root_common_splits
            if (expected_bitmask & split.bitmask) == expected_bitmask
        ]

        # If no non-root splits found, check if root split can contain it (fallback)
        is_fallback = False
        if not containing_splits and root_split:
            if (expected_bitmask & root_split.bitmask) == expected_bitmask:
                containing_splits = [root_split]
                is_fallback = True

        # Step 3: From the valid candidates, select the MINIMUM (smallest) split.
        if containing_splits:
            selected_split: Partition = min(
                containing_splits, key=lambda s: (s.bitmask.bit_count(), s.bitmask)
            )
            mapped_splits.append(selected_split)
            if is_fallback:
                fallback_count += 1
            else:
                success_count += 1
        else:
            # If no common split contains the expected taxa, fall back to root split.
            if root_split:
                mapped_splits.append(root_split)
                fallback_count += 1
            else:
                mapped_splits.append(
                    list(original_common_splits)[0] if original_common_splits else pivot_edge
                )
                fallback_count += 1

    if not jt_logger.disabled:
        jt_logger.info(
            f"✓ Mapped {len(mapped_splits)} pivots: "
            f"{success_count} internal, {fallback_count} root fallbacks"
        )

    return mapped_splits
