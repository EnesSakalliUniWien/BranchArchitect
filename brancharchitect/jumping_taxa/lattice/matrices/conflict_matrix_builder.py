"""
Conflict Matrix Construction
-----------------------------
Builds conflict matrices from pivot edge subproblems for the lattice algorithm.
Handles decision logic between nesting solutions and proper overlap conflicts.
"""

from __future__ import annotations
from typing import Dict

from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix
from brancharchitect.jumping_taxa.lattice.types.types import TopToBottom
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.matrices.conflict_collection import (
    collect_all_conflicts,
)
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    solution_size,
    matrix_row_size,
)
from brancharchitect.jumping_taxa.lattice.logging_helpers import (
    log_conflict_matrices,
    log_solution_comparison,
    log_solution_selection,
    log_nesting_only_solution,
    log_conflict_only_matrix,
)
from brancharchitect.logger import jt_logger


def _compute_conflict_taxa_size(conflicting_cover_pairs: PMatrix) -> int:
    """
    Compute the total taxa count from conflicting cover pairs.

    This helper function correctly counts taxa (not partitions) in the minimal
    cover of each conflict intersection. Used to compare conflict matrix size
    with nesting solution size during decision logic.

    Args:
        conflicting_cover_pairs: Matrix of conflicting PartitionSet pairs

    Returns:
        Total number of taxa across all minimal covers of conflicts
    """
    total = 0
    for left, right in conflicting_cover_pairs:
        minimal = (left & right).minimum_cover()
        total += sum(len(p.taxa) for p in minimal)  # Count TAXA, not partitions
    return total


def build_conflict_matrix(
    lattice_edge: PivotEdgeSubproblem,
) -> PMatrix:
    """
    Computes conflicting pairs of covers between two trees and returns them as a matrix.

    Each row in the returned matrix contains a conflicting pair [t1_cover, t2_cover].

    Decision logic: If nesting solutions are found, compares them with the conflict matrix
    and returns the approach that yields the smallest solution (most parsimonious).

    Args:
        lattice_edge: A PivotEdgeSubproblem object containing frontier information from both trees

    Returns:
        A matrix (list of lists) of conflicting PartitionSet pairs, or a 1×1 matrix
        containing the smallest nesting solution if that's more parsimonious.
        If no conflicts are found, returns an empty list.
    """
    left_covers: Dict[Partition, TopToBottom] = lattice_edge.tree1_child_frontiers
    right_covers: Dict[Partition, TopToBottom] = lattice_edge.tree2_child_frontiers

    # Collect all conflict types from cover pairs
    conflicting_cover_pairs, nesting_solutions, bottom_matrix = collect_all_conflicts(
        left_covers, right_covers
    )

    if not jt_logger.disabled:
        log_conflict_matrices(bottom_matrix, conflicting_cover_pairs)

    # DECISION LOGIC: Choose between nesting solutions and conflict matrix
    # Compare the size of solutions to select the most parsimonious approach
    if nesting_solutions and conflicting_cover_pairs:
        # Both nesting and proper overlap conflicts exist - choose smaller solution

        # Find the row with smallest size
        min_row_size = min(matrix_row_size(row) for row in bottom_matrix)

        # Filter solutions from rows with smallest size
        candidates_with_smallest_rows = [
            (sol, idx)
            for idx, sol in enumerate(nesting_solutions)
            if matrix_row_size(bottom_matrix[idx]) == min_row_size
        ]

        # Among those, select the one with smallest solution size
        smallest_nesting_solution = min(
            candidates_with_smallest_rows, key=lambda x: solution_size(x[0])
        )[0]

        nesting_size = solution_size(smallest_nesting_solution)

        # For conflict matrix, we would compute intersections which typically
        # yield smaller solutions than the original covers
        # Use consistent metric: count taxa (not partitions) for fair comparison
        conflict_size_estimate = _compute_conflict_taxa_size(conflicting_cover_pairs)

        if not jt_logger.disabled:
            log_solution_comparison(nesting_size, min_row_size, conflict_size_estimate)

        # If conflict matrix would yield smaller solution, use it
        # In case of tie (equal sizes), prefer nesting solution (simpler logic)
        if conflict_size_estimate < nesting_size:
            if not jt_logger.disabled:
                log_solution_selection("conflict", nesting_size, conflict_size_estimate)
            return conflicting_cover_pairs
        else:
            if not jt_logger.disabled:
                log_solution_selection("nesting", nesting_size, conflict_size_estimate)
            return [[smallest_nesting_solution]]

    # Only nesting solutions exist - return solution from smallest row
    elif nesting_solutions:
        # Find the row with smallest size
        min_row_size = min(matrix_row_size(row) for row in bottom_matrix)

        # Filter solutions from rows with smallest size
        candidates_with_smallest_rows = [
            (sol, idx)
            for idx, sol in enumerate(nesting_solutions)
            if matrix_row_size(bottom_matrix[idx]) == min_row_size
        ]

        # Among those, select the one with smallest solution size
        smallest_nesting = min(
            candidates_with_smallest_rows, key=lambda x: solution_size(x[0])
        )[0]

        if not jt_logger.disabled:
            log_nesting_only_solution(
                len(nesting_solutions), min_row_size, smallest_nesting
            )

        # Return as 1×1 matrix: [[solution]]
        # When generalized_meet_product processes a 1×1 square matrix,
        # it returns the matrix element directly without computing intersection
        return [[smallest_nesting]]

    # Only conflict matrix exists - return it
    if not jt_logger.disabled:
        log_conflict_only_matrix(conflicting_cover_pairs)
    return conflicting_cover_pairs
