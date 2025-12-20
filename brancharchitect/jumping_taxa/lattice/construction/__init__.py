"""
Lattice Construction
====================

Functions for building the lattice structure including pivot lattices,
child frontiers, and cover relations.
"""

from brancharchitect.jumping_taxa.lattice.construction.build_pivot_lattices import (
    construct_sublattices,
    build_conflict_matrix,
)
from brancharchitect.jumping_taxa.lattice.construction.child_frontiers import (
    compute_child_frontiers,
)
from brancharchitect.jumping_taxa.lattice.construction.cover_relations import (
    are_covers_incomparable,
    has_nesting_relationship,
    collect_all_conflicts,
)
from brancharchitect.jumping_taxa.lattice.construction.logging_helpers import (
    log_lattice_construction_start,
    log_pivot_processing,
    log_lattice_edge_details,
    log_conflict_matrices,
    log_solution_comparison,
    log_solution_selection,
    log_nesting_only_solution,
    log_conflict_only_matrix,
)

__all__ = [
    # Lattice building
    "construct_sublattices",
    "build_conflict_matrix",
    # Child frontiers
    "compute_child_frontiers",
    # Cover relations
    "are_covers_incomparable",
    "has_nesting_relationship",
    "collect_all_conflicts",
    # Logging helpers
    "log_lattice_construction_start",
    "log_pivot_processing",
    "log_lattice_edge_details",
    "log_conflict_matrices",
    "log_solution_comparison",
    "log_solution_selection",
    "log_nesting_only_solution",
    "log_conflict_only_matrix",
]
