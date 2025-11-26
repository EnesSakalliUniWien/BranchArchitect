"""
Logging Helpers for Lattice Construction
-----------------------------------------
Dedicated logging functions for build_pivot_lattices module.
Extracted to improve code organization and reusability.
"""

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.types import PMatrix
from brancharchitect.jumping_taxa.lattice.meet_product_solvers import solution_size
from brancharchitect.jumping_taxa.debug import jt_logger


def log_lattice_construction_start(
    t1: Node, t2: Node, intersection: PartitionSet[Partition]
) -> None:
    """Log initial lattice construction information."""
    jt_logger.info(f"Found {len(intersection)} common splits to process")
    jt_logger.compare_tree_splits(tree1=t1, tree2=t2)


def log_pivot_processing(pivot_split: Partition) -> None:
    """Log processing of individual pivot split."""
    jt_logger.info(f"Processing pivot_split {pivot_split.bipartition()} in both trees")


def log_lattice_edge_details(edge: PivotEdgeSubproblem) -> None:
    """Log detailed HTML tables for a lattice edge."""
    jt_logger.log_lattice_edge_tables(
        edge,
        show_common_covers=True,
        show_unique_min_covers=True,
        show_atoms=True,
        tablefmt="html",
    )


def log_conflict_matrices(
    bottom_matrix: PMatrix, conflicting_cover_pairs: PMatrix
) -> None:
    """Log conflict matrices for debugging."""
    if bottom_matrix:
        jt_logger.matrix(
            bottom_matrix, title="Bottoms Conflict Matrix (Nesting Relationships)"
        )
    if conflicting_cover_pairs:
        jt_logger.matrix(conflicting_cover_pairs, title="Covering Conflict Matrix")


def log_solution_comparison(
    nesting_size: int, min_row_size: int, conflict_size_estimate: int
) -> None:
    """Log comparison between nesting and conflict solutions."""
    jt_logger.info(
        f"Comparison: Nesting solution size={nesting_size} (from row size {min_row_size}), "
        f"Conflict matrix size estimate={conflict_size_estimate}"
    )


def log_solution_selection(
    selected: str, nesting_size: int, conflict_size: int
) -> None:
    """Log which solution type was selected."""
    if selected == "conflict":
        jt_logger.info(
            f"Selecting conflict matrix (smaller: {conflict_size} < {nesting_size})"
        )
    else:
        jt_logger.info(
            f"Selecting nesting solution (smaller or equal: {nesting_size} <= {conflict_size})"
        )


def log_nesting_only_solution(
    num_solutions: int, min_row_size: int, smallest_nesting: PartitionSet[Partition]
) -> None:
    """Log selection of nesting-only solution."""
    jt_logger.info(
        f"Found {num_solutions} nesting solution(s), "
        f"selected from smallest row (size {min_row_size}) "
        f"with solution size {solution_size(smallest_nesting)}: {smallest_nesting}"
    )


def log_conflict_only_matrix(conflicting_cover_pairs: PMatrix) -> None:
    """Log conflict-only matrix."""
    jt_logger.matrix(conflicting_cover_pairs, title="Partition Conflict Matrix")
