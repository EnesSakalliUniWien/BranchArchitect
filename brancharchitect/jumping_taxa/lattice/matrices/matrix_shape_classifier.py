"""Matrix shape classification for partition-matrix solving."""

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix


class MatrixCategory(Enum):
    """
    Categories of matrix shapes for solving strategy selection.

    Each category maps to a specific solving strategy:
    - VECTOR: Simple intersection of two partition sets
    - SQUARE: Diagonal intersection strategy (main & counter diagonals)
    - RECTANGULAR: Row-wise intersection, return all non-empty results
    - UNSUPPORTED: Unknown or unsupported shape
    """

    VECTOR = auto()  # 1×2: Simple intersection
    SQUARE = auto()  # n×n: Diagonal strategy
    RECTANGULAR = auto()  # k×2: Row-wise strategy
    UNSUPPORTED = auto()  # Unknown/unsupported shape


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
