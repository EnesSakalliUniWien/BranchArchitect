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


@dataclass(slots=True)
class TopToBottom:
    """
    A data structure to link top-level shared splits (frontiers) to their
    corresponding bottom-level splits within a child's subtree.
    """

    shared_top_splits: PartitionSet[Partition]
    bottom_to_frontiers: dict[Partition, PartitionSet[Partition]]
