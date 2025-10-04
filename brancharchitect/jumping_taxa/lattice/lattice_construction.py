from __future__ import annotations
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet, count_full_overlaps
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.lattice.matrix_ops import PMatrix
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
    format_set,
)
from typing import (
    Optional,
    List,
    Callable,
)  # Added TypingSet, TypeVar, cast
from brancharchitect.elements.partition import Partition

"""
Lattice Construction Utilities
-----------------------------
This module provides the main logic and helpers for constructing the lattice of reticulation events
between two phylogenetic trees, including split/cover/partition analysis and edge depth propagation.
"""


# ============================================================================
# Lattice Construction API (Entry Point)
# ============================================================================
def construct_sub_lattices(left_tree: Node, right_tree: Node) -> List[LatticeEdge]:
    """Compute detailed split information for two trees."""
    # Ensure both trees have their indices built

    # Rebuild split indices to ensure consistency after taxa deletion
    left_tree.build_split_index()
    right_tree.build_split_index()

    left_splits: PartitionSet[Partition] = left_tree.to_splits()  # fresh splits
    right_splits: PartitionSet[Partition] = right_tree.to_splits()  # fresh splits

    jt_logger.info(f"Tree splits after potential taxa deletion:")
    jt_logger.info(f"  Left tree has {len(left_splits)} splits")
    jt_logger.info(f"  Right tree has {len(right_splits)} splits")

    # Get common splits and verify they exist in both trees
    union_splits: PartitionSet[Partition] = left_splits.intersection(right_splits)

    if not union_splits:
        jt_logger.info(
            "No common splits found between trees - terminating lattice construction"
        )
        return []

    jt_logger.info(f"Found {len(union_splits)} common splits to process")
    lattice_edges: List[LatticeEdge] = []

    jt_logger.compare_tree_splits(tree1=left_tree, tree2=right_tree)

    for split in union_splits:
        if len(split) == 1:
            continue

        left_node: Node | None = left_tree.find_node_by_split(split)

        right_node: Node | None = right_tree.find_node_by_split(split)

        # Ensure nodes are found before proceeding
        if left_node is None or right_node is None:
            continue

        # For shared_splits_with_leaves, we need all splits including leaves, so bypass cache intentionally
        shared_splits_with_leaves: PartitionSet[Partition] = left_node.to_splits(
            with_leaves=True
        ) & right_node.to_splits(with_leaves=True)

        left_child_splits: PartitionSet[Partition] = get_child_splits(left_node)

        right_child_splits: PartitionSet[Partition] = get_child_splits(right_node)

        common_child_splits: PartitionSet[Partition] = (
            left_child_splits & right_child_splits
        )

        has_unique_child_splits: bool = (
            left_child_splits != common_child_splits
            or common_child_splits != right_child_splits
        )

        # Process further if there are child splits unique to either the left or right tree.
        if has_unique_child_splits:
            shared_splits_with_leaves.discard(split)

            jt_logger.info(
                f"Processing common split {split.bipartition()} in both trees"
            )

            left_common_covers: list[PartitionSet[Partition]] = compute_cover_elements(
                left_node, left_child_splits, shared_splits_with_leaves
            )
            right_common_covers: list[PartitionSet[Partition]] = compute_cover_elements(
                right_node, right_child_splits, shared_splits_with_leaves
            )

            left_unique_partition_sets: list[PartitionSet[Partition]] = compute_unique(
                left_node, right_node, left_child_splits, lambda ps: ps
            )
            right_unique_partition_sets: list[PartitionSet[Partition]] = compute_unique(
                right_node, left_node, right_child_splits, lambda ps: ps
            )

            # Assign s_edge_depth based on tree depth to all descendants
            # Add 1 to ensure internal nodes have non-zero s_edge_depth values

            lattice_edges.append(
                LatticeEdge(
                    split=split,
                    t1_common_covers=left_common_covers,
                    t2_common_covers=right_common_covers,
                    child_meet=common_child_splits,
                    left_node=left_node,
                    right_node=right_node,
                    encoding=left_tree.taxa_encoding,
                    t1_unique_partition_sets=left_unique_partition_sets,
                    t2_unique_partition_sets=right_unique_partition_sets,
                )
            )

    return lattice_edges


###############################################################################
# Core Lattice Construction Logic
###############################################################################
def get_child_splits(node: Node) -> PartitionSet[Partition]:
    """
    Compute the set of child splits for a node n.

    Definition:
      If n has children { c₁, c₂, …, cₖ }, then D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    # Ensure node._encoding is not None by using "or {}"
    return PartitionSet(
        {child.split_indices for child in node.children},
        encoding=node.taxa_encoding or {},
    )


###############################################################################
# Utility Functions
###############################################################################


def compute_cover_elements(
    parent_node: Node,
    child_split_set: PartitionSet[Partition],
    shared_splits: PartitionSet[Partition],
) -> List[PartitionSet[Partition]]:
    """
    For each child split of a node, compute its covering element.

    Cover(child) = (child.to_splits(with_leaves=True) ∩ common_excluding).cover()

    Returns a PartitionSet of cover elements.
    """
    cover_elements: List[PartitionSet[Partition]] = []
    for split in child_split_set:
        child_node = parent_node.find_node_by_split(split)
        if child_node is not None:
            # For cover computation, we need all splits including leaves, so bypass cache intentionally
            node_splits: PartitionSet[Partition] = child_node.to_splits(
                with_leaves=True
            )
            cover_candidate: PartitionSet[Partition] = node_splits & shared_splits
            covering_element: PartitionSet[Partition] = cover_candidate.cover()
            cover_elements.append(covering_element)
        else:
            raise ValueError(f"Split {split} not found in tree {parent_node}")
    # Add safety check for arms
    if not cover_elements:
        raise ValueError(f"Arms not found for split {parent_node.split_indices} ")
    return cover_elements


def compute_unique(
    target_node: Node,
    reference_node: Node,
    target_child_splits: PartitionSet[Partition],
    transform: Callable[[PartitionSet[Partition]], PartitionSet[Partition]],
) -> List[PartitionSet[Partition]]:
    """
    Generalized unique computation for atoms, covers, or partition sets.
    """
    unique_elements: List[PartitionSet[Partition]] = []
    reference_node_splits: PartitionSet[Partition] = (
        reference_node.to_splits()
    )  # uses cache
    for split in target_child_splits:
        child_node = target_node.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet[Partition] = child_node.to_splits()  # uses cache
            unique_splits_target: PartitionSet[Partition] = (
                node_splits - reference_node_splits
            )
            unique_elements.append(transform(unique_splits_target))
    return unique_elements


def build_partition_conflict_matrix(lattice_edge: LatticeEdge) -> Optional[PMatrix]:
    """
    Identifies conflicting pairs of covers between two trees and returns them as a matrix.

    Each row in the returned matrix contains a conflicting pair [t1_cover, t2_cover].

    Args:
        lattice_edge: A LatticeEdge object containing cover information from both trees

    Returns:
        A matrix (list of lists) of conflicting PartitionSet pairs, or None if
        no conflicts are found or if cover lists are empty.
    """

    left_covers = lattice_edge.t1_common_covers
    right_covers = lattice_edge.t2_common_covers

    # Early return if either cover list is empty
    if not left_covers or not right_covers:
        jt_logger.warning("Cannot build matrix with empty cover lists.")
        return None

    jt_logger.info(
        f"Building Partition Conflict Matrix for edge {format_set(set(lattice_edge.split))} "
        f"({len(left_covers)} left covers vs {len(right_covers)} right covers)"
    )

    conflicting_cover_pairs: PMatrix = []

    # Check for singleton covers (special case handling)
    left_singleton_index: int | None = find_first_singleton_cover_index(left_covers)
    right_singleton_index: int | None = find_first_singleton_cover_index(right_covers)

    # Case 1: Both trees have singleton covers
    if left_singleton_index is not None and right_singleton_index is not None:
        return _handle_both_trees_with_singletons(
            left_covers,
            right_covers,
            lattice_edge.t1_unique_partition_sets,
            lattice_edge.t2_unique_partition_sets,
        )

    # Case 2: Asymmetric singleton case - one tree has only singletons, other has compound covers
    if (
        left_singleton_index is None
        and right_singleton_index is not None
        and all(len(cover) == 1 for cover in right_covers)
    ) or (
        right_singleton_index is None
        and left_singleton_index is not None
        and all(len(cover) == 1 for cover in left_covers)
    ):
        for left_cover in left_covers:
            for right_cover in right_covers:
                intersection = left_cover & right_cover
                if intersection:
                    conflicting_cover_pairs.append([left_cover, right_cover])

        # Display the asymmetric singleton case matrix
        if conflicting_cover_pairs:
            jt_logger.matrix(
                conflicting_cover_pairs, title="Asymmetric Singleton Case Matrix"
            )

        return conflicting_cover_pairs

    # Case 3: General case - find all conflicting pairs
    for left_cover in left_covers:
        for right_cover in right_covers:
            intersection: PartitionSet[Partition] = left_cover & right_cover
            left_minus_right: PartitionSet[Partition] = left_cover - right_cover
            right_minus_left: PartitionSet[Partition] = right_cover - left_cover

            # A conflict exists when all three sets are non-empty
            if intersection and left_minus_right and right_minus_left:
                conflicting_cover_pairs.append([left_cover, right_cover])

    # Display the resulting conflict matrix if it's not empty
    if conflicting_cover_pairs:
        jt_logger.matrix(conflicting_cover_pairs, title="Partition Conflict Matrix")

    return conflicting_cover_pairs


def _handle_both_trees_with_singletons(
    left_covers: list[PartitionSet[Partition]],
    right_covers: list[PartitionSet[Partition]],
    left_unique_partition_sets: list[PartitionSet[Partition]],
    right_unique_partition_sets: list[PartitionSet[Partition]],
) -> PMatrix:
    """
    Handle the case when both trees have singleton covers.

    Args:
        left_covers: List of PartitionSet objects for the left tree.
        right_covers: List of PartitionSet objects for the right tree.
        left_unique_partition_sets: List of unique PartitionSets for the left tree.
        right_unique_partition_sets: List of unique PartitionSets for the right tree.

    Returns:
        PMatrix: List of conflicting cover pairs.
    """
    conflicting_cover_pairs: PMatrix = []

    # Find singleton covers using the modularized helper
    left_singleton = find_singleton_cover(left_covers)
    right_singleton = find_singleton_cover(right_covers)

    if left_singleton is None or right_singleton is None:
        raise ValueError("No singleton PartitionSet found in one or both cover lists.")

    jt_logger.info(
        f"Singletons found in both trees: {format_set(set(left_singleton))} and {format_set(set(right_singleton))}"
    )

    # Get the first partition from each singleton (since singleton is a PartitionSet of length 1)
    left_partition: Partition = next(iter(left_singleton))
    right_partition: Partition = next(iter(right_singleton))

    # Find conflicting non-singleton PartitionSets using the modularized helper
    left_conflict = find_conflicting_cover_index(left_covers, right_singleton)
    right_conflict = find_conflicting_cover_index(right_covers, left_singleton)

    if left_conflict is None or right_conflict is None:
        raise ValueError(
            "No conflicting singleton PartitionSet found in one or both cover lists."
        )

    left_conflict_index, left_conflict_set = left_conflict
    right_conflict_index, right_conflict_set = right_conflict

    left_overlap_count = count_full_overlaps(
        left_partition, right_unique_partition_sets[right_conflict_index]
    )
    right_overlap_count = count_full_overlaps(
        right_partition, left_unique_partition_sets[left_conflict_index]
    )

    # Detailed overlap mapping and matrix analysis
    jt_logger.section("=== OVERLAP ANALYSIS & MATRIX MAPPING ===")
    jt_logger.info(f"Left singleton: {format_set(set(left_singleton))}")
    jt_logger.info(f"Left partition (indices): {left_partition}")
    jt_logger.info(f"Right conflict set: {format_set(set(right_conflict_set))}")
    jt_logger.info(f"Right partition (indices): {right_partition}")
    jt_logger.info(f"Left conflict set: {format_set(set(left_conflict_set))}")
    jt_logger.info(f"Right singleton: {format_set(set(right_singleton))}")

    jt_logger.info(
        f"Left overlap target: {format_set(set(right_unique_partition_sets[right_conflict_index]))}"
    )
    jt_logger.info(
        f"Right overlap target: {format_set(set(left_unique_partition_sets[left_conflict_index]))}"
    )
    jt_logger.info(f"Left overlap count: {left_overlap_count}")
    jt_logger.info(f"Right overlap count: {right_overlap_count}")

    # Show both possible matrices
    jt_logger.info("=== MATRIX OPTIONS ===")
    jt_logger.info(
        f"Option A: [{format_set(set(left_singleton))}, {format_set(set(right_conflict_set))}]"
    )
    jt_logger.info(
        f"Option B: [{format_set(set(left_conflict_set))}, {format_set(set(right_singleton))}]"
    )

    # Show what each matrix intersection would produce
    intersection_A = left_singleton & right_conflict_set
    intersection_B = left_conflict_set & right_singleton
    jt_logger.info(
        f"Option A intersection: {format_set(set(intersection_A))} (size: {len(intersection_A)})"
    )
    jt_logger.info(
        f"Option B intersection: {format_set(set(intersection_B))} (size: {len(intersection_B)})"
    )

    if left_overlap_count > right_overlap_count:
        jt_logger.info(
            f"=== DECISION: Choose Option A (overlap {left_overlap_count} > {right_overlap_count}) ==="
        )
        conflicting_cover_pairs.append([left_singleton, right_conflict_set])
        jt_logger.info(
            f"Matrix will be: [{format_set(set(left_singleton))}, {format_set(set(right_conflict_set))}]"
        )
        jt_logger.info(f"Vector meet will produce: {format_set(set(intersection_A))}")
    elif left_overlap_count < right_overlap_count:
        jt_logger.info(
            f"=== DECISION: Choose Option B (overlap {left_overlap_count} < {right_overlap_count}) ==="
        )
        conflicting_cover_pairs.append([left_conflict_set, right_singleton])
        jt_logger.info(
            f"Matrix will be: [{format_set(set(left_conflict_set))}, {format_set(set(right_singleton))}]"
        )
        jt_logger.info(f"Vector meet will produce: {format_set(set(intersection_B))}")
    else:  # Equal overlaps
        jt_logger.info(
            f"=== DECISION: Equal overlaps ({left_overlap_count}), create 2x2 matrix ==="
        )
        conflicting_cover_pairs.append([left_conflict_set, right_singleton])
        conflicting_cover_pairs.append([left_singleton, right_conflict_set])
        jt_logger.info("2x2 Matrix:")
        jt_logger.info(
            f"  Row 1: [{format_set(set(left_conflict_set))}, {format_set(set(right_singleton))}]"
        )
        jt_logger.info(
            f"  Row 2: [{format_set(set(left_singleton))}, {format_set(set(right_conflict_set))}]"
        )
        jt_logger.info(f"  Row 1 intersection: {format_set(set(intersection_B))}")
        jt_logger.info(f"  Row 2 intersection: {format_set(set(intersection_A))}")

    jt_logger.info("=== END OVERLAP ANALYSIS ===\n")

    # Display the resulting matrix using the consistent matrix function
    if conflicting_cover_pairs:
        jt_logger.matrix(
            conflicting_cover_pairs, title="Singleton Cover Conflict Matrix"
        )

    return conflicting_cover_pairs


def find_first_singleton_cover_index(
    list_of_covers: List[PartitionSet[Partition]],
) -> Optional[int]:
    """
    Finds the index of the first PartitionSet in a list that represents a singleton.

    A singleton cover is defined as a PartitionSet containing exactly one Partition,
    and that Partition itself contains exactly one index.

    Args:
        covers: A list of PartitionSet objects to search through.

    Returns:
        The integer index of the first singleton PartitionSet found, or None if
        no singleton is found in the list.
    """
    for index, covers in enumerate(list_of_covers):
        # Check if the PartitionSet has exactly one Partition
        if len(covers) == 1:
            return index  # Found a singleton, return its index
    return None  # No singleton found in the list


def find_singleton_cover(
    covers: list[PartitionSet[Partition]],
) -> Optional[PartitionSet[Partition]]:
    """
    Returns the first singleton PartitionSet from a list of PartitionSet objects.

    Args:
        covers: A list of PartitionSet objects to search through.

    Returns:
        The first singleton PartitionSet found in the list, or None if not found.
    """
    for cover in covers:
        if len(cover) == 1:
            return cover
    return None


def find_conflicting_cover_index(
    covers: list[PartitionSet[Partition]],
    singleton: PartitionSet[Partition],
) -> Optional[tuple[int, PartitionSet[Partition]]]:
    """
    Finds the first PartitionSet in covers that contains the singleton PartitionSet as a subset.

    Args:
        covers: List of PartitionSet objects to search through.
        singleton: The singleton PartitionSet to check for as a subset.

    Returns:
        Tuple of (index, PartitionSet) if found, else None.
    """
    for i, reference_partition_set in enumerate(covers):
        if singleton.issubset(reference_partition_set):
            return i, reference_partition_set
    return None


def are_cover_lists_equivalent(
    list1: List[PartitionSet[Partition]], list2: List[PartitionSet[Partition]]
) -> bool:
    """
    Checks if two lists of PartitionSet objects contain the same elements,
    ignoring order and handling potential duplicates.

    Args:
        list1: The first list of PartitionSet objects.
        list2: The second list of PartitionSet objects.

    Returns:
        True if the lists contain the same PartitionSets (ignoring order),
        False otherwise.
    """
    from collections import Counter

    return Counter(list1) == Counter(list2)
