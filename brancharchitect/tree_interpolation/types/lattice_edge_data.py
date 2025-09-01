"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)


class LatticeEdgeData:
    """Structured data for lattice edge processing."""

    def __init__(
        self, edges: List[Partition], solutions: Dict[Partition, List[List[Partition]]]
    ):
        self.edges: List[Partition] = edges
        self.solutions: Dict[Partition, List[List[Partition]]] = solutions
        self.target_depths: Dict[Partition, float] = {}
        self.reference_depths: Dict[Partition, float] = {}

    def compute_depths(self, target: Node, reference: Node) -> None:
        """Compute and store depth mappings for both trees."""
        # Import here to avoid circular dependency

        self.target_depths = compute_lattice_edge_depths(self.edges, target)
        self.reference_depths = compute_lattice_edge_depths(self.edges, reference)

    def get_sorted_edges(
        self,
        use_reference: bool = False,
        ascending: bool = False,
        secondary_sort_key: Optional[Callable[[Partition], Any]] = None,
    ) -> List[Partition]:
        """
        Get s-edges sorted by depth, with optional secondary sorting.
        Args:
            use_reference: If True, sort by reference tree depths; otherwise, use target tree depths.
            ascending: If True, sort from smallest to largest depth (leaves to root).
            secondary_sort_key: An optional function to use as a secondary sort key.

        Returns:
            A list of sorted Partition objects.
        """
        depth_map = self.reference_depths if use_reference else self.target_depths

        # Create a stable sorting key. The primary key is depth.
        # The secondary key, if provided, resolves ties.
        # A tertiary key (the partition itself) ensures fully deterministic sorting.
        def sort_key(p: Partition):
            primary = depth_map.get(p, 0)
            secondary = secondary_sort_key(p) if secondary_sort_key else 0
            # Sorting by the partition representation makes the sort fully deterministic
            return (primary, secondary, p)

        return sorted(
            self.edges,
            key=sort_key,
            reverse=not ascending,
        )
