"""
Lattice Solvers
===============

Matrix solving algorithms for the lattice algorithm.
Includes meet product computation and pivot edge solving.

"""

from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    split_matrix,
    union_split_matrix_results,
    generalized_meet_product,
)
from brancharchitect.jumping_taxa.lattice.matrices.matrix_shape_classifier import (
    MatrixClassifier,
    MatrixCategory,
)
from .lattice_solver import LatticeSolver

__all__ = [
    # Meet product solvers
    "split_matrix",
    "union_split_matrix_results",
    "generalized_meet_product",
    # Matrix classification
    "MatrixClassifier",
    "MatrixCategory",
    # Pivot edge solving
    "LatticeSolver",
]
