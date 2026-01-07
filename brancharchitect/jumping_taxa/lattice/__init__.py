"""Lattice package.

Provides the lattice algorithm for computing jumping taxa between phylogenetic trees.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
        LatticeSolver,
    )
    from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix
    from brancharchitect.jumping_taxa.lattice.types.child_frontiers import (
        ChildFrontiers,
    )
    from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
        PivotEdgeSubproblem,
    )
    from brancharchitect.jumping_taxa.lattice.types.registry import (
        SolutionRegistry,
        compute_solution_rank_key,
    )
    from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
        sort_pivot_edges_by_subset_hierarchy,
        topological_sort_edges,
    )

__all__ = [
    "LatticeSolver",
    "PMatrix",
    "ChildFrontiers",
    "PivotEdgeSubproblem",
    "SolutionRegistry",
    "compute_solution_rank_key",
    "sort_pivot_edges_by_subset_hierarchy",
    "topological_sort_edges",
]

_LAZY_IMPORTS = {
    "LatticeSolver": "brancharchitect.jumping_taxa.lattice.solvers.lattice_solver",
    "PMatrix": "brancharchitect.jumping_taxa.lattice.matrices.types",
    "ChildFrontiers": "brancharchitect.jumping_taxa.lattice.types.child_frontiers",
    "PivotEdgeSubproblem": "brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem",
    "SolutionRegistry": "brancharchitect.jumping_taxa.lattice.types.registry",
    "compute_solution_rank_key": "brancharchitect.jumping_taxa.lattice.types.registry",
    "sort_pivot_edges_by_subset_hierarchy": "brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering",
    "topological_sort_edges": "brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering",
}


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
