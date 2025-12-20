"""
Lattice Solvers
===============

Matrix solving algorithms for the lattice algorithm.
Includes meet product computation and pivot edge solving.

Note: Uses lazy imports to avoid circular dependencies.
"""

__all__ = [
    # Meet product solvers
    "split_matrix",
    "union_split_matrix_results",
    "generalized_meet_product",
    "solution_size",
    "matrix_row_size",
    # Matrix classification
    "MatrixClassifier",
    "MatrixCategory",
    "RowClassifier",
    "RowType",
    # Pivot edge solving
    "solve_pivot_edges",
    "lattice_algorithm",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    # Meet product solvers
    if name in (
        "split_matrix",
        "union_split_matrix_results",
        "generalized_meet_product",
        "solution_size",
        "matrix_row_size",
    ):
        from brancharchitect.jumping_taxa.lattice.solvers import meet_product_solvers

        return getattr(meet_product_solvers, name)
    # Matrix classification
    elif name in ("MatrixClassifier", "MatrixCategory", "RowClassifier", "RowType"):
        from brancharchitect.jumping_taxa.lattice.solvers import matrix_shape_classifier

        return getattr(matrix_shape_classifier, name)
    # Pivot edge solving
    elif name in ("solve_pivot_edges", "lattice_algorithm"):
        from brancharchitect.jumping_taxa.lattice.solvers import pivot_edge_solver

        return getattr(pivot_edge_solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
