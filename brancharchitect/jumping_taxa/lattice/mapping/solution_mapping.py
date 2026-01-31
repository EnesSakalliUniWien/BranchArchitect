"""
Solution Mapping: maps pivot edge solutions from pruned to original trees.
"""

from typing import Dict, List

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.mapping.iterative_pivot_mappings import (
    map_iterative_pivot_edges_to_original,
)
from brancharchitect.logger import jt_logger


def map_solutions_to_original_trees(
    solutions_dict: Dict[Partition, List[Partition]],
    original_tree1: Node,
    original_tree2: Node,
) -> Dict[Partition, List[Partition]]:
    """
    Map pivot edges from pruned trees to their corresponding splits in original trees.

    Args:
        solutions_dict: Flat solution partitions keyed by pivot edges from pruned trees
        original_tree1: Original unpruned tree 1
        original_tree2: Original unpruned tree 2

    Returns:
        Flat solution partitions keyed by pivot edges mapped to original trees
    """
    if not jt_logger.disabled:
        jt_logger.info("[lattice] Mapping pivot edges to original trees...")

    pivot_edges_list = list(solutions_dict.keys())
    solutions_list = [solutions_dict[pivot] for pivot in pivot_edges_list]

    mapped_pivot_edges = map_iterative_pivot_edges_to_original(
        pivot_edges_list,
        original_tree1,
        original_tree2,
        solutions_list,
    )

    # Build new dictionary with mapped pivot edges, aggregating solutions for collisions
    mapped_solutions_dict: Dict[Partition, List[Partition]] = {}
    for pivot_edge, mapped_pivot in zip(pivot_edges_list, mapped_pivot_edges):
        if mapped_pivot not in mapped_solutions_dict:
            mapped_solutions_dict[mapped_pivot] = []
        mapped_solutions_dict[mapped_pivot].extend(solutions_dict[pivot_edge])

    if not jt_logger.disabled:
        jt_logger.info(
            f"[lattice] Mapped {len(pivot_edges_list)} pivot edges to original trees"
        )

    return mapped_solutions_dict


def map_solutions_to_common_subtrees(
    solutions_dict: Dict[Partition, List[Partition]],
    original_tree1: Node,
    original_tree2: Node,
) -> Dict[Partition, List[Partition]]:
    mapped: Dict[Partition, List[Partition]] = {}
    for pivot_edge, solutions in solutions_dict.items():
        # Optimization: Use PartitionSet for automatic global uniqueness and efficient hashing
        # This replaces the manual `seen_bitmasks` loop.
        mapped_set = PartitionSet(encoding=original_tree1.taxa_encoding)

        for solution in solutions:
            mapped_parts = _map_solution_partition_to_common_subtrees(
                solution,
                pivot_edge,
                original_tree1,
                original_tree2,
            )
            mapped_set.update(mapped_parts)

        # Sort combined results for determinism
        mapped[pivot_edge] = sorted(
            mapped_set, key=lambda p: (len(p.indices), p.bitmask)
        )
    return mapped


def _map_solution_partition_to_common_subtrees(
    partition: Partition,
    pivot_edge: Partition,
    original_tree1: Node,
    original_tree2: Node,
) -> List[Partition]:
    total_taxa = len(original_tree1.taxa_encoding)

    # Strict checks: Solution cannot be the Root or the Pivot Edge itself
    if partition.bitmask.bit_count() == total_taxa:
        raise ValueError("Cannot map root partition as solution")

    # Note: We compare bitmasks to check if they represent the same set of taxa
    if partition.bitmask == pivot_edge.bitmask:
        raise ValueError("Cannot return pivot edge as solution")

    # Strict Check: Solution must be strictly contained under the pivot edge
    # This prevents cross-pivot leakage where a solution might include excluded taxa
    if (partition.bitmask & pivot_edge.bitmask) != partition.bitmask:
        raise ValueError(
            f"Solution partition {partition} is not strictly contained under pivot edge {pivot_edge}. "
            "Leakage detected."
        )

    # If the solution exists as a split in both original trees, keep it as-is.
    try:
        node_in_t1 = original_tree1.find_node_by_split(partition)
        node_in_t2 = original_tree2.find_node_by_split(partition)
        if node_in_t1 is not None and node_in_t2 is not None:
            return [partition]
    except ValueError:
        # Encoding mismatch; fall back to common-split mapping below.
        pass

    # Primary Strategy: Covering splits (minimal supersets under the pivot subtree)
    covers = _find_covering_common_splits(
        partition, pivot_edge, original_tree1, original_tree2
    )

    if covers:
        return covers

    # Failure (Strict: Do not fall back to Pivot Edge or Root)
    raise ValueError(
        f"Could not map solution partition {partition} to any common structure under pivot {pivot_edge}. "
        "No exact match found and decomposition failed."
    )


def _find_covering_common_splits(
    partition: Partition,
    pivot_edge: Partition,
    tree1: Node,
    tree2: Node,
) -> List[Partition]:
    """
    Find minimal covering common splits (MRCAs) for the solution partition.

    Two-phase algorithm:
    1. First, try to find minimal supersets (single MRCA that contains the partition)
    2. If no superset exists, find maximal subsets (multiple MRCAs that partition the solution)

    Uses the 'exclude' parameter to strictly forbid the Pivot Edge from being returned.
    If no covering common splits exist, falls back to singleton leaves for the partition.
    """
    if not jt_logger.disabled:
        jt_logger.debug(
            f"[cover-map] start partition={partition} pivot_edge={pivot_edge}"
        )

    exclusion_set = {pivot_edge}

    pivot_edge_subtree_t1 = tree1.find_node_by_split(pivot_edge)
    pivot_edge_subtree_t2 = tree2.find_node_by_split(pivot_edge)

    if pivot_edge_subtree_t1 is None or pivot_edge_subtree_t2 is None:
        if not jt_logger.disabled:
            jt_logger.debug(
                f"[cover-map] missing pivot subtree t1={pivot_edge_subtree_t1 is None} "
                f"t2={pivot_edge_subtree_t2 is None}"
            )
        raise ValueError("Pivot edge not found in both trees")

    pivot_edge_subtree_t1_splits = pivot_edge_subtree_t1.to_splits(with_leaves=True)
    pivot_edge_subtree_t2_splits = pivot_edge_subtree_t2.to_splits(with_leaves=True)

    common_pivot_edge_subtree_splits = (
        pivot_edge_subtree_t1_splits & pivot_edge_subtree_t2_splits
    )

    if not jt_logger.disabled:
        jt_logger.debug(
            f"[cover-map] subtree splits t1={len(pivot_edge_subtree_t1_splits)} "
            f"t2={len(pivot_edge_subtree_t2_splits)} common={len(common_pivot_edge_subtree_splits)}"
        )

    # Phase 1: Try to find minimal supersets (single MRCA containing partition)
    cover_set = common_pivot_edge_subtree_splits.minimals_over(
        partition, exclude=exclusion_set
    )

    if cover_set:
        if not jt_logger.disabled:
            cover_indices = cover_set.resolve_to_indices()
            jt_logger.debug(
                f"[cover-map] cover_set size={len(cover_set)} covers={cover_indices}"
            )
        return sorted(
            cover_set.fast_partitions, key=lambda s: (len(s.indices), s.bitmask)
        )

    # Phase 2: No superset found - find maximal subsets that cover the partition
    subset_set = common_pivot_edge_subtree_splits.maximals_under(partition)

    if subset_set:
        if not jt_logger.disabled:
            jt_logger.debug(
                f"[cover-map] fallback to maximal subsets: {len(subset_set)} subsets"
            )
        return sorted(
            subset_set.fast_partitions, key=lambda s: (len(s.indices), s.bitmask)
        )

    # Phase 3: Worst-case fallback - map to individual leaves
    if not jt_logger.disabled:
        jt_logger.debug(
            f"[cover-map] fallback to leaves indices={list(partition.indices)}"
        )
    return sorted(
        [
            Partition.from_bitmask(1 << idx, tree1.taxa_encoding)
            for idx in partition.indices
        ],
        key=lambda s: (len(s.indices), s.bitmask),
    )
