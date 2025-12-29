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
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


@dataclass(slots=True)
class TopToBottom:
    """
    A data structure to link top-level shared splits (frontiers) to their
    corresponding bottom-level splits within a child's subtree.
    """

    shared_top_splits: PartitionSet[Partition]
    bottom_to_frontiers: dict[Partition, PartitionSet[Partition]]

    def remove_partition(self, partition: Partition) -> None:
        """
        Safely remove a partition from both shared covers (frontiers) and internal mappings.

        This handles:
        1. Removing from the top-level shared splits set (covers).
        2. Removing from all frontier sets contained in the bottom-to-frontier mapping.
        3. Removing the partition itself if it acts as a bottom-level key.

        Using .pop() for key removal prevents Race/KeyErrors if checked separately.
        """
        # 1. Remove from covers (tops)
        self.shared_top_splits.discard(partition)

        # 2. Remove from frontier sets (values)
        for frontier_set in self.bottom_to_frontiers.values():
            frontier_set.discard(partition)

        # 3. Remove from bottom keys safely
        self.bottom_to_frontiers.pop(partition, None)
