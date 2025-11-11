"""
Matrix Classification Module
=============================

This module provides classification and analysis tools for partition matrices.
It separates the "what type is this matrix" logic from the "how to solve it" logic.

Key Responsibilities:
- Classify matrix shapes (vector, square, rectangular)
- Detect degenerate rows (singleton containment)
- Identify special cases (independent 2×2 matrices)
- Extract and separate different row types

This allows the solving algorithms to focus purely on computation while
classification logic remains testable and reusable.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brancharchitect.jumping_taxa.lattice.types import PMatrix, MatrixRow


class MatrixCategory(Enum):
    """
    Categories of matrix shapes for solving strategy selection.

    Each category maps to a specific solving strategy:
    - VECTOR: Simple intersection of two partition sets
    - SQUARE: Diagonal intersection strategy (main & counter diagonals)
    - RECTANGULAR: Row-wise intersection, return all non-empty results
    - DEGENERATE: All rows are degenerate (trivial solutions)
    - UNSUPPORTED: Unknown or unsupported shape
    """

    VECTOR = auto()  # 1×2: Simple intersection
    SQUARE = auto()  # n×n: Diagonal strategy
    RECTANGULAR = auto()  # k×2: Row-wise strategy
    DEGENERATE = auto()  # All rows are degenerate
    UNSUPPORTED = auto()  # Unknown/unsupported shape


class RowType(Enum):
    """
    Classification of individual matrix rows.

    DEGENERATE: A row where one side is a singleton partition set contained
                in the other side. Example: [{p1}, {p1, p2, p3}]
                These represent trivial conflicts with obvious solutions.

    STANDARD: Normal conflicting pairs that require lattice solving.
    """

    DEGENERATE = auto()  # Singleton contained in other side
    STANDARD = auto()  # Normal conflicting pair


class RowClassifier:
    """
    Classifies individual matrix rows.

    Provides methods to determine if a row represents a degenerate case
    (where one partition set is a singleton contained in the other).
    """

    @staticmethod
    def classify_row(row: "MatrixRow") -> RowType:
        """
        Determine if a row is degenerate or standard.

        A row is degenerate if:
        1. It has exactly 2 elements (left and right)
        2. One side is a singleton partition set
        3. That singleton is contained in the other side

        Args:
            row: A matrix row (list of PartitionSets)

        Returns:
            RowType.DEGENERATE if singleton containment detected,
            RowType.STANDARD otherwise
        """
        if len(row) != 2:
            return RowType.STANDARD

        left, right = row

        # Left singleton contained in right
        if len(left) == 1:
            try:
                single = next(iter(left))
                if single in right:
                    return RowType.DEGENERATE
            except StopIteration:
                pass

        # Right singleton contained in left
        if len(right) == 1:
            try:
                single = next(iter(right))
                if single in left:
                    return RowType.DEGENERATE
            except StopIteration:
                pass

        return RowType.STANDARD

    @staticmethod
    def is_degenerate(row: "MatrixRow") -> bool:
        """
        Check if row is degenerate (singleton contained in other side).

        Convenience method that returns a boolean instead of enum.

        Args:
            row: A matrix row to classify

        Returns:
            True if row is degenerate, False otherwise
        """
        return RowClassifier.classify_row(row) == RowType.DEGENERATE


class MatrixClassifier:
    """
    Classifies matrices and determines solving strategy.
    """

    @staticmethod
    def classify_matrix(matrix: "PMatrix") -> MatrixCategory:
        """
        Classify matrix by shape for strategy selection.
        """
        if not matrix or not matrix[0]:
            return MatrixCategory.UNSUPPORTED

        rows = len(matrix)
        cols = len(matrix[0])

        # 1×2 vector
        if rows == 1 and cols == 2:
            return MatrixCategory.VECTOR

        # n×n square
        if rows == cols:
            return MatrixCategory.SQUARE

        # k×2 rectangular
        if cols == 2 and rows > 1:
            return MatrixCategory.RECTANGULAR

        return MatrixCategory.UNSUPPORTED

    @staticmethod
    def extract_degenerate_rows(matrix: "PMatrix") -> tuple["PMatrix", list["PMatrix"]]:
        """
        Separate degenerate rows from standard rows as 1×2 matrices.
        """
        base_rows: "PMatrix" = []
        degenerate_matrices: list["PMatrix"] = []

        for row in matrix:
            if RowClassifier.is_degenerate(row):
                degenerate_matrices.append([row])  # Wrap as 1×2 matrix
            else:
                base_rows.append(row)

        return base_rows, degenerate_matrices

    @staticmethod
    def is_independent_2x2(matrix: "PMatrix") -> bool:
        """Return True if 2×2 with different left column values (don’t split)."""
        if len(matrix) != 2 or len(matrix[0]) != 2:
            return False
        left_row_1 = frozenset(matrix[0][0])
        left_row_2 = frozenset(matrix[1][0])
        return left_row_1 != left_row_2

    @staticmethod
    def should_split(matrix: "PMatrix") -> bool:
        """
        Decide if matrix should be split into smaller matrices.
        """
        if MatrixClassifier.is_independent_2x2(matrix):
            return False
        if len(matrix) <= 1:
            return False
        if not matrix or not matrix[0]:
            return False
        return True

    @staticmethod
    def validate_matrix(matrix: "PMatrix") -> tuple[int, int]:
        """
        Validate matrix structure and return (rows, cols).
        """
        if not matrix or not matrix[0]:
            raise ValueError(
                "Matrix must not be empty and must have at least one column."
            )
        rows, cols = len(matrix), len(matrix[0])
        if any(len(row) != cols for row in matrix):
            raise ValueError("All rows must have the same number of columns.")
        return rows, cols


def classify_matrix(matrix: "PMatrix") -> MatrixCategory:
    """Convenience: classify a matrix."""
    return MatrixClassifier.classify_matrix(matrix)
