"""
Conflict Matrix Collection: builds matrices of conflicting cover pairs for
lattice construction by iterating over frontier structures and applying
poset relation predicates.
"""

from __future__ import annotations
from itertools import product
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers
from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix
from brancharchitect.jumping_taxa.lattice.frontiers.poset_relations import (
    are_covers_incomparable,
    has_nesting_relationship,
    has_strict_containment_relationship,
    get_nesting_solution,
    get_top_containment_solutions,
)
from brancharchitect.logger import jt_logger


def collect_nesting_conflicts(
    left_child_frontiers: ChildFrontiers,
    right_child_frontiers: ChildFrontiers,
    nesting_solutions: list[PartitionSet[Partition]],
    bottom_matrix: PMatrix,
) -> None:
    """
    Collect nesting relationships (A ⊆ B ∨ B ⊆ A) between bottom sets.

    For each pair of bottom sets from both frontier structures:
    1. Skip empty sets (vacuously true but meaningless)
    2. If nesting exists, append smaller set to nesting_solutions
    3. Log the pair in bottom_matrix (indices synchronized with nesting_solutions)

    Example: {(X),(A1)} ⊆ {(X),(A1),(B4)} → solution = {(X),(A1)}

    Side Effects:
        Appends to nesting_solutions and bottom_matrix when nesting is found.
    """
    for left_frontier_set, right_frontier_set in product(
        left_child_frontiers.bottom_partition_map.values(),
        right_child_frontiers.bottom_partition_map.values(),
    ):
        if not jt_logger.disabled:
            jt_logger.info(
                f"------Bottoms: {left_frontier_set} <-> {right_frontier_set}"
            )

        if not left_frontier_set or not right_frontier_set:
            continue

        if has_nesting_relationship(left_frontier_set, right_frontier_set):
            solution = get_nesting_solution(left_frontier_set, right_frontier_set)
            # Keep indices synchronized for bottom-nesting diagnostics.
            nesting_solutions.append(solution)
            bottom_matrix.append([left_frontier_set, right_frontier_set])

    if not jt_logger.disabled:
        jt_logger.matrix(
            bottom_matrix, title="Bottoms Conflict Matrix (Nesting Relationships)"
        )


def collect_top_cover_containment_conflict(
    left_cover: PartitionSet[Partition],
    right_cover: PartitionSet[Partition],
    top_containment_solutions: list[PartitionSet[Partition]],
) -> None:
    """Collect a direct witness for strict containment between top covers."""
    if has_strict_containment_relationship(left_cover, right_cover):
        top_containment_solutions.extend(
            get_top_containment_solutions(left_cover, right_cover)
        )


def collect_all_conflicts(
    left_covers: dict[Partition, ChildFrontiers],
    right_covers: dict[Partition, ChildFrontiers],
) -> tuple[
    PMatrix,
    list[PartitionSet[Partition]],
    list[PartitionSet[Partition]],
    PMatrix,
]:
    """
    Collect all conflicts between two sets of cover frontiers.

    Conflict types:
        - Bottom nesting (A ⊆ B): direct minimal solution (smaller nested set)
        - Top-cover strict containment: direct alternatives for the frontier cut
        - Incomparability (proper overlap): requires meet product resolution

    For each (left_cover, right_cover) pair:
        1. Check bottoms for nesting → nesting_solutions
        2. Check tops for incomparability → conflicting_cover_pairs
        3. Check tops for strict containment → top_containment_solutions

    Returns:
        (conflicting_cover_pairs, nesting_solutions,
         top_containment_solutions, bottom_matrix)
        Note: bottom_matrix only describes bottom-nesting direct solutions.
    """
    conflicting_cover_pairs: PMatrix = []
    bottom_matrix: PMatrix = []
    nesting_solutions: list[PartitionSet[Partition]] = []
    top_containment_solutions: list[PartitionSet[Partition]] = []

    # Use itertools.product to generate all pairs of covers from both trees
    for left_child_frontiers, right_child_frontiers in product(
        left_covers.values(), right_covers.values()
    ):
        left_cover: PartitionSet[Partition] = left_child_frontiers.shared_top_splits
        right_cover: PartitionSet[Partition] = right_child_frontiers.shared_top_splits

        # Collect nesting relationships from bottom sets
        collect_nesting_conflicts(
            left_child_frontiers,
            right_child_frontiers,
            nesting_solutions,
            bottom_matrix,
        )

        collect_top_cover_containment_conflict(
            left_cover,
            right_cover,
            top_containment_solutions,
        )

        # Check for proper overlap (incomparable elements indicating conflict)
        if are_covers_incomparable(left_cover, right_cover):
            conflicting_cover_pairs.append([left_cover, right_cover])

    return (
        conflicting_cover_pairs,
        nesting_solutions,
        top_containment_solutions,
        bottom_matrix,
    )
