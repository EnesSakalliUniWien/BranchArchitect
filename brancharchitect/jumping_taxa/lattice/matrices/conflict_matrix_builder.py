"""
Conflict Matrix Construction
-----------------------------
Builds conflict matrices from pivot edge subproblems for the lattice algorithm.
Handles decision logic between nesting solutions and proper overlap conflicts.
"""

from __future__ import annotations
from typing import Dict

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.matrices.conflict_collection import (
    collect_all_conflicts,
)
from brancharchitect.jumping_taxa.lattice.logging_helpers import (
    log_conflict_matrices,
    log_conflict_only_matrix,
)
from brancharchitect.logger import jt_logger


def _nesting_solution_rank_key(
    solution: PartitionSet[Partition],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    sizes = tuple(sorted(partition.size for partition in solution))
    bitmasks = tuple(sorted(partition.bitmask for partition in solution))
    return (len(solution), sum(sizes), sizes, bitmasks)


def _deduplicate_direct_solutions(
    direct_solutions: list[PartitionSet[Partition]],
) -> list[PartitionSet[Partition]]:
    seen: set[tuple[int, ...]] = set()
    deduplicated: list[PartitionSet[Partition]] = []

    for solution in sorted(direct_solutions, key=_nesting_solution_rank_key):
        key = tuple(sorted(partition.bitmask for partition in solution))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(solution)

    return deduplicated


def _direct_solution_rows(
    direct_solutions: list[PartitionSet[Partition]],
) -> PMatrix:
    return [
        [direct_solution, direct_solution]
        for direct_solution in _deduplicate_direct_solutions(direct_solutions)
    ]


def _direct_solution_alternative_row(
    direct_solutions: list[PartitionSet[Partition]],
) -> PMatrix:
    rows = _direct_solution_rows(direct_solutions)
    return rows[:1]


def build_conflict_matrix(
    lattice_edge: PivotEdgeSubproblem,
    include_top_containment: bool = True,
) -> PMatrix:
    """
    Computes conflicting pairs of covers between two trees and returns them as a matrix.

    Each row in the returned matrix contains a conflicting pair [t1_cover, t2_cover].

    Bottom-nesting witnesses, top-cover containment witnesses, and proper-overlap
    rows are all local candidate sources. Direct witnesses are encoded as
    self-meet rows [S, S]; the solver treats those rows as already-solved
    candidates so they do not change the meet-product shape of overlap rows.

    Args:
        lattice_edge: A PivotEdgeSubproblem object containing frontier information from both trees
        include_top_containment: Whether to use strict top-cover containment
            rows. This is valid for freshly built tree frontiers, but not for
            residual cover probes created by mutating a pivot in place.

    Returns:
        A matrix (list of lists) containing proper-overlap rows plus direct
        self-meet candidate rows. If no conflicts are found, returns an empty list.
    """
    left_covers: Dict[Partition, ChildFrontiers] = lattice_edge.tree1_child_frontiers
    right_covers: Dict[Partition, ChildFrontiers] = lattice_edge.tree2_child_frontiers

    # Collect all conflict types from cover pairs
    (
        conflicting_cover_pairs,
        nesting_solutions,
        top_containment_solutions,
        bottom_matrix,
    ) = collect_all_conflicts(left_covers, right_covers)

    if not jt_logger.disabled:
        log_conflict_matrices(bottom_matrix, conflicting_cover_pairs)

    if conflicting_cover_pairs:
        matrix = conflicting_cover_pairs + _direct_solution_rows(nesting_solutions)
    elif nesting_solutions:
        matrix = _direct_solution_alternative_row(nesting_solutions)
    else:
        matrix = []

    if include_top_containment and top_containment_solutions:
        matrix += _direct_solution_rows(top_containment_solutions)

    if not matrix and not jt_logger.disabled:
        log_conflict_only_matrix(conflicting_cover_pairs)

    return matrix
