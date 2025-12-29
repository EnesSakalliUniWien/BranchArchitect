"""
Matrix Type Definitions
=======================

This module defines the types used for the conflict matrix structure in the lattice algorithm.
"""

from __future__ import annotations
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
