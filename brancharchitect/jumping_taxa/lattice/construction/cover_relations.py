"""
Cover Relations and Poset Comparisons
--------------------------------------
Functions for analyzing relationships between covers in the partial order (poset)
of phylogenetic splits. These relations determine conflict types and guide
lattice construction strategies.

Relations Analyzed:
  - Incomparability (proper overlap): Neither cover contains the other, but they overlap
  - Nesting (subset): One cover is contained within another
  - Disjointness: Covers share no common elements
"""

from __future__ import annotations
from itertools import product
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types.types import TopToBottom
from brancharchitect.jumping_taxa.lattice.types.types import PMatrix
from brancharchitect.jumping_taxa.debug import jt_logger


def are_covers_incomparable(
    left_cover: PartitionSet[Partition], right_cover: PartitionSet[Partition]
) -> bool:
    """
    Check if two covers are incomparable in the poset sense (have proper overlap).

    MATHEMATICAL DEFINITION:
        Three-Way Venn Decomposition - Returns True iff all three regions are non-empty:
        - A ∩ B ≠ ∅  (intersection)
        - A \\ B ≠ ∅  (left minus right)
        - B \\ A ≠ ∅  (right minus left)

    PHYLOGENETIC INTERPRETATION:
        Incomparable elements in the partial order indicate that the covers share
        some splits but neither contains the other. This represents a phylogenetic
        conflict requiring reticulation to resolve.

    COMPARISON WITH OTHER RELATIONS:
        - If one cover contains the other: nesting relationship (comparable, no conflict)
        - If covers are disjoint: no relationship (independent, no conflict)
        - If proper overlap exists: incomparable elements (CONFLICT)

    POSET THEORY:
        In a partially ordered set (poset), two elements a and b are incomparable
        if neither a ≤ b nor b ≤ a. For sets under subset ordering, this translates
        to: ¬(A ⊆ B) ∧ ¬(B ⊆ A) ∧ A ∩ B ≠ ∅

    Args:
        left_cover: Cover (shared top splits) from left tree
        right_cover: Cover (shared top splits) from right tree

    Returns:
        True if covers are incomparable (proper overlap, indicating conflict),
        False otherwise (nesting, disjoint, or identical)

    Examples:
        >>> # Incomparable (proper overlap - CONFLICT)
        >>> left = PartitionSet([A, B, C])
        >>> right = PartitionSet([B, C, D])
        >>> are_covers_incomparable(left, right)  # True: {A}, {B,C}, {D} all non-empty

        >>> # Nesting (no conflict)
        >>> left = PartitionSet([A, B])
        >>> right = PartitionSet([A, B, C])
        >>> are_covers_incomparable(left, right)  # False: left ⊆ right

        >>> # Disjoint (no conflict)
        >>> left = PartitionSet([A, B])
        >>> right = PartitionSet([C, D])
        >>> are_covers_incomparable(left, right)  # False: A ∩ B = ∅
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

    MATHEMATICAL DEFINITION:
        Returns True iff: left ⊆ right ∨ right ⊆ left
        (One set is a subset of the other in the partial order)

    PHYLOGENETIC INTERPRETATION:
        A nesting relationship indicates that one clade is contained within
        another, which necessitates a jumping taxon to reconcile the two
        tree topologies.

    POSET THEORY:
        In a partially ordered set (poset), two elements a and b have a nesting
        relationship if a ≤ b or b ≤ a. For sets under subset ordering, this
        translates to: A ⊆ B ∨ B ⊆ A (comparable elements).

    Args:
        left_bottoms: Bottom split set from left tree
        right_bottoms: Bottom split set from right tree

    Returns:
        True if one bottom is nested within the other, False otherwise

    Examples:
        >>> # Nesting (left ⊆ right)
        >>> left = PartitionSet([A, B])
        >>> right = PartitionSet([A, B, C])
        >>> has_nesting_relationship(left, right)  # True: left ⊆ right

        >>> # Nesting (right ⊆ left)
        >>> left = PartitionSet([A, B, C])
        >>> right = PartitionSet([A])
        >>> has_nesting_relationship(left, right)  # True: right ⊆ left

        >>> # No nesting (incomparable)
        >>> left = PartitionSet([A, B])
        >>> right = PartitionSet([B, C])
        >>> has_nesting_relationship(left, right)  # False: neither is subset of other
    """
    return left_bottoms.issubset(right_bottoms) or right_bottoms.issubset(left_bottoms)


def get_nesting_solution(
    left_bottoms: PartitionSet[Partition], right_bottoms: PartitionSet[Partition]
) -> PartitionSet[Partition]:
    """
    For nesting relationships, return the smaller (nested) set as the solution.

    MATHEMATICAL RATIONALE:
        When A ⊆ B, the conflict is resolved by choosing elements from A
        (the smaller set), not by taking the intersection A ∩ B = A.

        This ensures that for pairs like [{(A1), (X)}, {(X)}]:
        - The nested set is {(X)} (since {(X)} ⊆ {(A1), (X)})
        - Return {(X)} as the solution
        - Later, {(A1)} will form another conflict and be resolved separately

    PHYLOGENETIC INTERPRETATION:
        The smaller set represents the minimal jumping taxon needed to
        resolve the immediate nesting conflict. Larger conflicts involving
        additional taxa will be discovered in subsequent iterations.

    OPTIMIZATION PRINCIPLE:
        By selecting the minimal (nested) set, we ensure:
        1. Most parsimonious solution (fewest taxa)
        2. Incremental conflict resolution (one nesting at a time)
        3. Proper ordering of conflict discovery

    Args:
        left_bottoms: Bottom split set from left tree
        right_bottoms: Bottom split set from right tree

    Returns:
        The smaller (nested) set, or the left set if they're equal

    Examples:
        >>> # Left nested in right
        >>> left = PartitionSet([A])
        >>> right = PartitionSet([A, B, C])
        >>> get_nesting_solution(left, right)  # Returns left (the smaller set)

        >>> # Right nested in left
        >>> left = PartitionSet([A, B, C])
        >>> right = PartitionSet([B])
        >>> get_nesting_solution(left, right)  # Returns right (the smaller set)
    """
    if left_bottoms.issubset(right_bottoms):
        # left is nested in right, return left (the smaller set)
        return left_bottoms
    elif right_bottoms.issubset(left_bottoms):
        # right is nested in left, return right (the smaller set)
        return right_bottoms
    else:
        # No nesting relationship - should not be called
        # Return intersection as fallback
        return left_bottoms & right_bottoms


def collect_nesting_conflicts(
    left_top_to_bottom: TopToBottom,
    right_top_to_bottom: TopToBottom,
    nesting_solutions: list[PartitionSet[Partition]],
    bottom_matrix: PMatrix,
) -> None:
    """
    Collect nesting relationships between bottom sets from two frontier structures.

    MATHEMATICAL FOUNDATION:
        In a partial order (poset), a nesting relationship occurs when:
            A ⊆ B  ∨  B ⊆ A  (one set is a subset of the other)

        For phylogenetic trees, this indicates a containment conflict where one
        clade is nested within another, requiring jumping taxa to reconcile.

    ALGORITHM:
        For each pair of bottom sets (left_bottoms, right_bottoms):
        1. Skip empty sets (∅ ⊆ anything is vacuously true but meaningless)
        2. Check if has_nesting_relationship(left_bottoms, right_bottoms)
        3. If nesting exists, compute the minimal solution:
           - Return the smaller (nested) set via get_nesting_solution()
           - Append to nesting_solutions for later selection
           - Log the pair in bottom_matrix for debugging

    PHYLOGENETIC INTERPRETATION:
        Bottom sets represent minimal shared splits under a cover. When one bottom
        is nested in another, it indicates that some taxa must "jump" to resolve
        the topological difference. The smaller nested set identifies the minimal
        jumping taxa.

    CARTESIAN PRODUCT:
        Uses itertools.product to generate all pairs (left_bottoms, right_bottoms)
        from the two frontier structures, equivalent to nested loops but more Pythonic.

    EXAMPLE:
        left_bottoms = {(X), (A1)}  ⊆  right_bottoms = {(X), (A1), (B4)}
        → Nesting detected: left ⊆ right
        → Solution: {(X), (A1)} (the smaller nested set)

    Args:
        left_top_to_bottom: Frontier structure from left tree child
        right_top_to_bottom: Frontier structure from right tree child
        nesting_solutions: Output list to collect nesting solutions
        bottom_matrix: Output list to log bottom pairs for debugging

    Side Effects:
        Appends to nesting_solutions and bottom_matrix when nesting is found
        NOTE: Indices in nesting_solutions and bottom_matrix are synchronized
    """
    # Use itertools.product to generate all pairs of bottom sets
    for left_bottoms, right_bottoms in product(
        left_top_to_bottom.bottom_to_frontiers.values(),
        right_top_to_bottom.bottom_to_frontiers.values(),
    ):
        if not jt_logger.disabled:
            jt_logger.info(f"------Bottoms: {left_bottoms} <-> {right_bottoms}")

        # Skip empty sets: ∅ ⊆ X is vacuously true but phylogenetically meaningless
        if not left_bottoms or not right_bottoms:
            continue

        # Check for nesting relationship: A ⊆ B or B ⊆ A
        if has_nesting_relationship(left_bottoms, right_bottoms):
            # Compute minimal solution: return the smaller (nested) set
            solution = get_nesting_solution(left_bottoms, right_bottoms)

            # IMPORTANT: Keep indices synchronized between nesting_solutions and bottom_matrix
            # nesting_solutions[i] corresponds to bottom_matrix[i]
            nesting_solutions.append(solution)
            bottom_matrix.append([left_bottoms, right_bottoms])

    if not jt_logger.disabled:
        jt_logger.matrix(
            bottom_matrix, title="Bottoms Conflict Matrix (Nesting Relationships)"
        )


def collect_all_conflicts(
    left_covers: dict[Partition, TopToBottom],
    right_covers: dict[Partition, TopToBottom],
) -> tuple[PMatrix, list[PartitionSet[Partition]], PMatrix]:
    """
    Collect all conflict types between two sets of cover frontiers.

    This function orchestrates the complete conflict collection process by:
    1. Iterating through all pairs of covers (Cartesian product)
    2. Detecting nesting conflicts in bottom sets
    3. Detecting incomparability conflicts in top sets (covers)

    CONFLICT TYPES:
        - Nesting conflicts: Bottom sets where one is contained in the other (A ⊆ B)
        - Incomparability conflicts: Top covers with proper overlap (neither contains the other)

    ALGORITHM:
        For each (left_cover, right_cover) pair:
        ├─ Extract top splits (covers) and bottom sets
        ├─ Check bottoms for nesting → add to nesting_solutions
        └─ Check tops for incomparability → add to conflicting_cover_pairs

    PHYLOGENETIC INTERPRETATION:
        The function identifies two distinct resolution strategies:
        - Nesting provides direct minimal solutions (the smaller nested set)
        - Incomparability requires intersection-based resolution (meet product)

    Args:
        left_covers: Frontier structures from left tree (T1 child frontiers)
        right_covers: Frontier structures from right tree (T2 child frontiers)

    Returns:
        A tuple of three elements:
        - conflicting_cover_pairs: Matrix rows of incomparable cover pairs
        - nesting_solutions: List of minimal nesting solutions
        - bottom_matrix: Matrix rows of nested bottom pairs (for debugging/selection)

    Synchronization:
        nesting_solutions[i] corresponds to bottom_matrix[i] (same index)
    """
    conflicting_cover_pairs: PMatrix = []
    bottom_matrix: PMatrix = []
    nesting_solutions: list[PartitionSet[Partition]] = []

    # Use itertools.product to generate all pairs of covers from both trees
    for left_top_to_bottom, right_top_to_bottom in product(
        left_covers.values(), right_covers.values()
    ):
        left_cover: PartitionSet[Partition] = left_top_to_bottom.shared_top_splits
        right_cover: PartitionSet[Partition] = right_top_to_bottom.shared_top_splits

        # Collect nesting relationships from bottom sets
        collect_nesting_conflicts(
            left_top_to_bottom,
            right_top_to_bottom,
            nesting_solutions,
            bottom_matrix,
        )

        # Check for proper overlap (incomparable elements indicating conflict)
        if are_covers_incomparable(left_cover, right_cover):
            conflicting_cover_pairs.append([left_cover, right_cover])

    return conflicting_cover_pairs, nesting_solutions, bottom_matrix
