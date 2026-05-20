"""
Poset Relations: pure predicates for comparing sets under subset ordering.

These functions determine relationships (nesting, incomparability, disjointness)
between phylogenetic split sets without any matrix construction logic.
"""

from __future__ import annotations
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition


def are_covers_incomparable(
    left_cover: PartitionSet[Partition], right_cover: PartitionSet[Partition]
) -> bool:
    """
    Check if two covers have proper overlap (incomparable in poset order).

    Three-Way Venn Test: Returns True iff all regions are non-empty:
        A ∩ B ≠ ∅, A \\ B ≠ ∅, B \\ A ≠ ∅

    Equivalently: ¬(A ⊆ B) ∧ ¬(B ⊆ A) ∧ A ∩ B ≠ ∅

    Phylogenetically, incomparability indicates conflict requiring reticulation.
    Nesting (A ⊆ B) or disjointness (A ∩ B = ∅) indicate no conflict.

    Examples:
        >>> are_covers_incomparable({A,B,C}, {B,C,D})  # True: proper overlap
        >>> are_covers_incomparable({A,B}, {A,B,C})    # False: nesting
        >>> are_covers_incomparable({A,B}, {C,D})      # False: disjoint
    """
    intersection: PartitionSet[Partition] = left_cover & right_cover
    left_minus_right: PartitionSet[Partition] = left_cover - right_cover
    right_minus_left: PartitionSet[Partition] = right_cover - left_cover
    return bool(intersection and left_minus_right and right_minus_left)


def has_nesting_relationship(
    left_bottoms: PartitionSet[Partition], right_bottoms: PartitionSet[Partition]
) -> bool:
    """
    Check if two bottom sets have a nesting (subset) relationship.

    Returns True iff: left ⊆ right ∨ right ⊆ left (comparable in poset order).
    Phylogenetically, nesting indicates one subtree contained in another,
    requiring jumping taxa to reconcile tree topologies.

    Examples:
        >>> has_nesting_relationship({A,B}, {A,B,C})  # True: left ⊆ right
        >>> has_nesting_relationship({A,B,C}, {A})    # True: right ⊆ left
        >>> has_nesting_relationship({A,B}, {B,C})    # False: incomparable
    """
    return left_bottoms.issubset(right_bottoms) or right_bottoms.issubset(left_bottoms)


def has_strict_containment_relationship(
    left_cover: PartitionSet[Partition], right_cover: PartitionSet[Partition]
) -> bool:
    """
    Check whether one frontier cover strictly contains the other.

    This is different from subtree nesting. The compared objects are covers of
    shared frontier atoms under two child boundaries. Strict containment means
    one side has all atoms from the other side plus at least one extra atom.
    """
    if not left_cover or not right_cover:
        return False
    return (
        left_cover.issubset(right_cover) and left_cover != right_cover
    ) or (
        right_cover.issubset(left_cover) and right_cover != left_cover
    )


def get_nesting_solution(
    left_bottoms: PartitionSet[Partition], right_bottoms: PartitionSet[Partition]
) -> PartitionSet[Partition]:
    """
    Return the smaller (nested) set as the solution for nesting conflicts.

    When A ⊆ B, choose A (the nested set) to ensure:
    - Parsimonious solution (minimal taxa)
    - Incremental resolution (larger conflicts discovered later)

    Examples:
        >>> get_nesting_solution({A}, {A,B,C})  # Returns {A} (nested set)
        >>> get_nesting_solution({A,B,C}, {B})  # Returns {B} (nested set)
    """
    # Symmetric solution: consistently return the intersection (Meet)
    # Since specific nesting is guaranteed by caller, this always returns the smaller set.
    return left_bottoms & right_bottoms


def get_top_containment_solutions(
    left_cover: PartitionSet[Partition], right_cover: PartitionSet[Partition]
) -> list[PartitionSet[Partition]]:
    """
    Return both witness sides of a strict top-cover containment.

    If A ⊂ B at the frontier-cover level, the boundary discrepancy is the
    binary cut A | (B \\ A). Either side can explain the move, so both are
    valid local candidates.
    """
    if left_cover.issubset(right_cover) and left_cover != right_cover:
        contained = left_cover
        complement = right_cover - left_cover
    elif right_cover.issubset(left_cover) and right_cover != left_cover:
        contained = right_cover
        complement = left_cover - right_cover
    else:
        return []

    solutions = [
        choice.maximal_elements()
        for choice in (contained, complement)
        if choice
    ]
    return sorted(solutions, key=_partition_set_rank_key)


def _partition_set_rank_key(
    partition_set: PartitionSet[Partition],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    sizes = tuple(sorted(partition.size for partition in partition_set))
    bitmasks = tuple(sorted(partition.bitmask for partition in partition_set))
    return (len(partition_set), sum(sizes), sizes, bitmasks)
