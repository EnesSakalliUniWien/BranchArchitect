"""
Lattice Type Definitions and Domain Model
==========================================

This module defines the core types used in phylogenetic tree lattice construction.
Understanding these types is essential to understanding the lattice algorithms.

Domain Concepts:
- **Partition**: A subset of taxa represented by indices (e.g., {0, 2, 3} for taxa A, C, D)
- **PartitionSet**: A collection of partitions that share the same taxon encoding
- **Split**: A bipartition of taxa that defines an edge in a phylogenetic tree
- **Cover**: A minimal set of partitions whose union equals a larger partition set
- **Lattice**: A partially ordered structure representing relationships between tree topologies

Mathematical Operations:
- **Meet (∧)**: Greatest lower bound in the lattice (implemented as intersection &)
- **Join (∨)**: Least upper bound in the lattice (implemented as union |)
- **Cover**: Minimal generating set for a partition collection
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

# ============================================================================
# Core Matrix Types
# ============================================================================

# A matrix element is a set of partitions
MatrixCell: TypeAlias = PartitionSet[Partition]

# A matrix row contains cells (typically 2 for conflict pairs)
MatrixRow: TypeAlias = list[MatrixCell]

# The partition matrix: rows of partition set pairs
# Used to represent conflicting cover pairs between two trees
PMatrix: TypeAlias = list[MatrixRow]

# Example PMatrix structure:
# [
#   [PartitionSet({p1, p2}), PartitionSet({p3, p4})],  # Row 1: conflict pair
#   [PartitionSet({p1, p5}), PartitionSet({p3, p6})],  # Row 2: conflict pair
# ]


# ============================================================================
# Solution Types
# ============================================================================

# A single lattice solution is a set of partitions
LatticeSolution: TypeAlias = PartitionSet[Partition]

# Multiple alternative solutions (e.g., from different diagonals)
LatticeSolutions: TypeAlias = list[LatticeSolution]


# ============================================================================
# Tree-Specific Types
# ============================================================================

# Cover elements for a single tree's subtree
# Each PartitionSet represents the cover of one child's splits
TreeCovers: TypeAlias = list[PartitionSet[Partition]]

# Unique partition sets from one tree not present in another
# Used to identify structural differences between trees
UniquePartitionSets: TypeAlias = list[PartitionSet[Partition]]


# ============================================================================
# Matrix Dimension Categories
# ============================================================================


class MatrixShape:
    """
    Defines the different matrix shapes and their solving strategies.

    - Vector (1×2): Simple intersection of two partition sets
    - Square (n×n): Diagonal intersection strategy (main & counter diagonals)
    - Rectangular (k×2): Row-wise intersection, return all non-empty results
    """

    @staticmethod
    def is_vector(matrix: PMatrix) -> bool:
        """1×2 matrix - single pair of partition sets to intersect."""
        return len(matrix) == 1 and len(matrix[0]) == 2

    @staticmethod
    def is_square(matrix: PMatrix) -> bool:
        """n×n matrix - use diagonal intersection strategy."""
        if not matrix or not matrix[0]:
            return False
        return len(matrix) == len(matrix[0])

    @staticmethod
    def is_rectangular_two_column(matrix: PMatrix) -> bool:
        """k×2 matrix where k > 1 - use row-wise intersection."""
        return len(matrix) > 1 and all(len(row) == 2 for row in matrix)


# ============================================================================
# Pairing Strategy Types
# ============================================================================


class PairingStrategy:
    """
    Defines how solutions from different matrices are combined.

    Reverse Mapping (for 2 matrices with equal results):
        Position i in Matrix 1 pairs with position (n-1-i) in Matrix 2
        Example: [0,1,2] pairs with [2,1,0]
        Creates complementary dependency relationships

    Union Strategy (for unequal or >2 matrices):
        Take union of results at each position across all matrices
        Fallback when reverse mapping cannot be applied
    """

    REVERSE_MAPPING = "reverse_mapping"
    UNION = "union"


# ============================================================================
# Helper Functions
# ============================================================================


def describe_matrix(matrix: PMatrix) -> str:
    """
    Return a human-readable description of a matrix's structure.

    Example output:
        "3×2 rectangular matrix with 3 conflict pairs"
        "2×2 square matrix (diagonal strategy applicable)"
        "1×2 vector (simple intersection)"
    """
    if not matrix or not matrix[0]:
        return "empty matrix"

    rows = len(matrix)
    cols = len(matrix[0])

    if MatrixShape.is_vector(matrix):
        return f"1×2 vector (simple intersection)"
    elif MatrixShape.is_square(matrix):
        return f"{rows}×{cols} square matrix (diagonal strategy applicable)"
    elif MatrixShape.is_rectangular_two_column(matrix):
        return f"{rows}×2 rectangular matrix with {rows} conflict pairs"
    else:
        return f"{rows}×{cols} matrix (unsupported shape)"


def count_non_empty_cells(matrix: PMatrix) -> int:
    """Count how many cells in the matrix contain non-empty partition sets."""
    return sum(1 for row in matrix for cell in row if cell)


def matrix_stats(matrix: PMatrix) -> dict:
    """
    Return statistics about a matrix for debugging/logging.

    Returns:
        dict with keys: rows, cols, non_empty_cells, total_cells,
                       shape_description, strategy
    """
    if not matrix or not matrix[0]:
        return {
            "rows": 0,
            "cols": 0,
            "non_empty_cells": 0,
            "total_cells": 0,
            "shape_description": "empty",
            "strategy": None,
        }

    rows = len(matrix)
    cols = len(matrix[0])
    total_cells = rows * cols
    non_empty = count_non_empty_cells(matrix)

    # Determine strategy
    if MatrixShape.is_vector(matrix):
        strategy = "vector_meet"
    elif MatrixShape.is_square(matrix):
        strategy = "diagonal_meet"
    elif MatrixShape.is_rectangular_two_column(matrix):
        strategy = "row_wise_meet"
    else:
        strategy = "unsupported"

    return {
        "rows": rows,
        "cols": cols,
        "non_empty_cells": non_empty,
        "total_cells": total_cells,
        "shape_description": describe_matrix(matrix),
        "strategy": strategy,
    }


@dataclass
class TopToBottom:
    """
    A data structure to link top-level shared splits (frontiers) to their
    corresponding bottom-level splits within a child's subtree.
    """

    shared_top_splits: PartitionSet[Partition]
    bottom_to_frontiers: dict[Partition, PartitionSet[Partition]]
