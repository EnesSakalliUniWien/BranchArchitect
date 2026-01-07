"""
Lattice Type Definitions
========================

Core types and data structures for the lattice algorithm.
"""

from brancharchitect.jumping_taxa.lattice.matrices.types import (
    MatrixCell,
    MatrixRow,
    PMatrix,
)
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
    get_child_splits,
)
from brancharchitect.jumping_taxa.lattice.types.registry import (
    SolutionRegistry,
    compute_solution_rank_key,
)

__all__ = [
    "MatrixCell",
    "MatrixRow",
    "PMatrix",
    "ChildFrontiers",
    "PivotEdgeSubproblem",
    "get_child_splits",
    "SolutionRegistry",
    "compute_solution_rank_key",
]
