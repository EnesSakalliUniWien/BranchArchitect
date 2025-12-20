"""Lattice package.

Provides the lattice algorithm for computing jumping taxa between phylogenetic trees.

Package Structure:
- types/: Core data types (PMatrix, TopToBottom, PivotEdgeSubproblem)
- solvers/: Matrix solving algorithms (meet products, pivot edge solving)
- construction/: Lattice building (child frontiers, cover relations)
- orchestration/: High-level drivers (iterative algorithm, taxa deletion)
- ordering/: Edge ordering algorithms (topological sort, depth ordering)
- mapping/: Solution mapping utilities

Primary Entry Point:
    from brancharchitect.jumping_taxa.lattice.orchestration import compute_pivot_solutions_with_deletions

Note: Submodules are NOT imported at package level to prevent circular imports.
Import submodules directly where needed.
"""

__all__ = [
    # Primary entry point (import directly from orchestration)
    "compute_pivot_solutions_with_deletions",
    # Types (import directly from types)
    "PMatrix",
    "TopToBottom",
    "PivotEdgeSubproblem",
    "SolutionRegistry",
    "compute_solution_rank_key",
    # Ordering (import directly from ordering)
    "sort_pivot_edges_by_subset_hierarchy",
    "topological_sort_edges",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "compute_pivot_solutions_with_deletions":
        from brancharchitect.jumping_taxa.lattice.orchestration.compute_pivot_solutions_with_deletions import (
            compute_pivot_solutions_with_deletions,
        )

        return compute_pivot_solutions_with_deletions
    elif name == "PMatrix":
        from brancharchitect.jumping_taxa.lattice.types.types import PMatrix

        return PMatrix
    elif name == "TopToBottom":
        from brancharchitect.jumping_taxa.lattice.types.types import TopToBottom

        return TopToBottom
    elif name == "PivotEdgeSubproblem":
        from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
            PivotEdgeSubproblem,
        )

        return PivotEdgeSubproblem
    elif name == "SolutionRegistry":
        from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry

        return SolutionRegistry
    elif name == "compute_solution_rank_key":
        from brancharchitect.jumping_taxa.lattice.types.registry import (
            compute_solution_rank_key,
        )

        return compute_solution_rank_key
    elif name == "sort_pivot_edges_by_subset_hierarchy":
        from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
            sort_pivot_edges_by_subset_hierarchy,
        )

        return sort_pivot_edges_by_subset_hierarchy
    elif name == "topological_sort_edges":
        from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
            topological_sort_edges,
        )

        return topological_sort_edges
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
