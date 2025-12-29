"""
Matrices: conflict matrix construction, collection, and solving for lattice algorithm.

This module handles building, classifying, and solving conflict matrices from
phylogenetic split covers and frontier structures.
"""

# Import matrix solving first (no circular dependencies)
from brancharchitect.jumping_taxa.lattice.matrices.matrix_shape_classifier import (
    MatrixClassifier,
    MatrixCategory,
)
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    split_matrix,
    union_split_matrix_results,
    generalized_meet_product,
    solution_size,
    matrix_row_size,
)

# Import conflict collection and building (depends on above)
from brancharchitect.jumping_taxa.lattice.matrices.conflict_collection import (
    collect_all_conflicts,
    collect_nesting_conflicts,
)
from brancharchitect.jumping_taxa.lattice.matrices.conflict_matrix_builder import (
    build_conflict_matrix,
)

__all__ = [
    # Conflict collection
    "collect_all_conflicts",
    "collect_nesting_conflicts",
    # Matrix building
    "build_conflict_matrix",
    # Matrix solving
    "split_matrix",
    "union_split_matrix_results",
    "generalized_meet_product",
    "solution_size",
    "matrix_row_size",
    # Matrix classification
    "MatrixClassifier",
    "MatrixCategory",
]
