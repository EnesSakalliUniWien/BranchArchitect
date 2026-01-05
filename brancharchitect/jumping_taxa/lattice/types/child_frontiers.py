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
class ChildFrontiers:
    """
    Links shared splits (frontiers) to unique splits (bottoms) within a subtree.

    Mathematical Context:
        Given pivot edge P in both trees T₁ and T₂:
        - **shared_top_splits**: Maximal elements of (T₁.splits ∩ T₂.splits) under P
        - **bottom_partition_map**: Maps each unique split u ∈ (T₁ \\ T₂) to the
          shared splits it covers: {f ∈ shared_top_splits | f ⊆ u}

    Used by the conflict matrix builder to identify nesting relationships
    and proper overlap conflicts between tree topologies.

    Attributes:
        shared_top_splits: Frontier splits shared by both trees (maximal common clades).
        bottom_partition_map: Maps unique splits → covered frontier splits.
    """

    shared_top_splits: PartitionSet[Partition]
    bottom_partition_map: dict[Partition, PartitionSet[Partition]]

    def remove_partition(self, partition: Partition) -> None:
        """
        Safely remove a partition and any intersecting partitions from covers and mappings.

        This handles:
        1. Removing any partition in shared_top_splits that intersects with the given partition.
        2. Removing any intersecting partition from frontier sets in bottom_partition_map.
        3. Removing the partition itself if it acts as a bottom-level key.
        """
        partition_mask = partition.bitmask

        # 1. Remove intersecting from covers (tops)
        if partition in self.shared_top_splits:
            self.shared_top_splits.discard(partition)

        # 2. Remove intersecting from frontier sets (values)
        for frontier_set in self.bottom_partition_map.values():
            if partition in frontier_set:
                frontier_set.discard(partition)

        # 3. Remove from bottom keys safely
        keys_to_remove = [
            k
            for k in self.bottom_partition_map.keys()
            if (k.bitmask & partition_mask) != 0
        ]
        for k in keys_to_remove:
            self.bottom_partition_map.pop(k, None)
