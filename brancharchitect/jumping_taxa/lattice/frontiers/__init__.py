"""
Frontiers: pivot lattice building, child frontier computation, and poset relations.
"""

from brancharchitect.jumping_taxa.lattice.frontiers.build_pivot_lattices import (
    construct_sublattices,
)
from brancharchitect.jumping_taxa.lattice.frontiers.child_frontiers import (
    compute_child_frontiers,
)
from brancharchitect.jumping_taxa.lattice.frontiers.poset_relations import (
    are_covers_incomparable,
    has_nesting_relationship,
    get_nesting_solution,
)
from brancharchitect.jumping_taxa.lattice.logging_helpers import (
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
    # Child frontiers
    "compute_child_frontiers",
    # Poset relations (pure predicates)
    "are_covers_incomparable",
    "has_nesting_relationship",
    "get_nesting_solution",
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
