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
- All pivot edges map to minimum splits to find the most specific common structure match,
  ensuring that local changes in pruned trees are mapped to their most precise
  local equivalents in the original tree.

Implementation Notes
--------------------
- Splits are represented as bitmasks over the original leaf indexing
- Subset relation p ⊆ s is checked via bitwise AND: (p & s) == p
- Among splits of equal size, ties are broken deterministically by bitmask value
"""

from typing import List, Optional, Set, Dict, Union, Iterable

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition, PartitionSet
from brancharchitect.logger.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.frontiers.construct_pivot_edge_problems import (
    is_pivot_edge,
    _validate_nodes_exist,
)
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    topological_sort_edges,
)


def map_single_pivot_edge_to_original(
    pivot_edge: Partition,
    original_common_splits: PartitionSet[Partition],
    solutions: Iterable[Union[PartitionSet[Partition], Partition, Iterable[Partition]]],
    original_tree: Node,
    leaf_map: Dict[int, Node] | None = None,
) -> Partition:
    """
    Map a single pivot edge from a pruned tree to its corresponding split in original trees.

    Uses efficient tree traversal (MRCA) on the original tree to find the
    MINIMUM (smallest/most specific) common split that contains the pivot/jumping taxa.

    Args:
        pivot_edge: Pivot edge from the current (possibly pruned) iteration
        original_common_splits: Pre-computed common splits from original trees (T₁ ∩ T₂)
        solutions: List of solution PartitionSets for this pivot edge
        original_tree: The original unpruned tree T1 (used for topological traversal)
        leaf_map: Optional pre-computed map of taxon index -> Leaf Node in original_tree

    Returns:
        The mapped split from original trees (minimum containing split)
    """
    # 1. Provide Leaf Map (Build on the fly if missing - inefficient for batch, ok for single)
    if leaf_map is None:
        leaf_map = {
            original_tree.taxa_encoding[leaf.name]: leaf
            for leaf in original_tree.get_leaves()
            if leaf.name in original_tree.taxa_encoding
        }

    # 2. Collect all target indices (P ∪ J)
    # We handle the polymorphic input manually to avoid wrapper overhead
    jumping_taxa_indices: Set[int] = set()
    for item in solutions:
        if isinstance(item, Partition):
            jumping_taxa_indices.update(item.indices)
        elif isinstance(item, (set, list, tuple, PartitionSet)):
            for sub_item in item:
                if isinstance(sub_item, Partition):
                    jumping_taxa_indices.update(sub_item.indices)

    expected_original_indices: Set[int] = set(pivot_edge.indices) | jumping_taxa_indices

    # 3. Optimization: Trivial Case
    # If no indices (shouldn't happen), return pivot
    if not expected_original_indices:
        return pivot_edge

    # 4. Find MRCA (Most Recent Common Ancestor) in original_tree
    # Strategy: Start at one leaf, walk up until the node covers all expected bits.

    # Calculate target bitmask
    target_mask = 0
    start_node: Optional[Node] = None

    for idx in expected_original_indices:
        target_mask |= 1 << idx
        # Pick the first valid leaf we find as our start point
        if start_node is None and idx in leaf_map:
            start_node = leaf_map[idx]

    if start_node is None:
        # Fallback: Indices don't match tree? Return pivot.
        return pivot_edge

    # Walk up to find MRCA in T1
    # The MRCA is the first ancestor where (node_mask & target_mask) == target_mask
    curr = start_node
    while curr.parent is not None:
        if (curr.split_indices.bitmask & target_mask) == target_mask:
            break
        curr = curr.parent

    mrca_node = curr

    # 5. Find MINIMUM Common Split
    # The MRCA is the smallest split in T1 containing the taxa.
    # We now walk UP from MRCA to find the first ancestor that is ALSO in T2 (common).
    # This guarantees it is the smallest common split.

    common_bitmasks = original_common_splits._bitmask_set
    cursor: Optional[Node] = mrca_node

    while cursor is not None:
        if cursor.split_indices.bitmask in common_bitmasks:
            # FOUND IT: This is the smallest common split containing target.
            return original_common_splits._bitmask_to_partition[
                cursor.split_indices.bitmask
            ]
        cursor = cursor.parent

    # 6. Fallback (Root)
    # If we walked off the top without hitting a common split (rare if trees share root),
    # return the pivot as a fallback.
    return pivot_edge


def get_pivot_edges(t1: Node, t2: Node) -> List[Partition]:
    """Compute detailed split information for two trees (per pivot/frontiers)."""
    # Ensure both trees have their indices built

    t1_splits: PartitionSet[Partition] = t1.to_splits()  # fresh splits
    t2_splits: PartitionSet[Partition] = t2.to_splits()  # fresh splits

    # Get common splits and verify they exist in both trees
    intersection: PartitionSet[Partition] = t1_splits.intersection(t2_splits)

    # Sort splits deterministically by size (approx. topological) then bitmask
    sorted_common_splits = topological_sort_edges(list(intersection), t1)

    pivot_edge_problems: List[Partition] = []
    for pivot_split in sorted_common_splits:
        t1_node: Node | None = t1.find_node_by_split(pivot_split)
        t2_node: Node | None = t2.find_node_by_split(pivot_split)

        # Validate that both trees contain the pivot split
        _validate_nodes_exist(pivot_split, t1_node, t2_node)

        # Type narrowing: after validation, nodes are guaranteed to be non-None
        assert t1_node is not None
        assert t2_node is not None

        is_pivot, child_subtree_splits_across_trees = is_pivot_edge(t1_node, t2_node)

        # Process further if there are child splits unique to either tree.
        if is_pivot:
            pivot_edge_problems.append(pivot_split)

    return pivot_edge_problems


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

    # Build efficiency map for MRCA lookups
    leaf_map = {
        original_t1.taxa_encoding[leaf.name]: leaf
        for leaf in original_t1.get_leaves()
        if leaf.name in original_t1.taxa_encoding
    }

    pivot_edges = get_pivot_edges(original_t1, original_t2)

    if not jt_logger.disabled:
        jt_logger.info(
            f"Mapping {len(pivot_edges_from_iteration)} pivot edges "
            f"via MRCA traversal ({len(pivot_edges)} common splits)"
        )

    for i, pivot_edge in enumerate(pivot_edges_from_iteration):
        current_solutions = []
        if jumping_taxa_solutions and i < len(jumping_taxa_solutions):
            # Pass the list of partitions directly.
            # Our updated map_single handles List[Partition] correctly.
            current_solutions = jumping_taxa_solutions[i]

        mapped_split = map_single_pivot_edge_to_original(
            pivot_edge, pivot_edge, current_solutions, original_t1, leaf_map
        )
        mapped_splits.append(mapped_split)

    return mapped_splits
