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
        self,
        edges: List[Partition],
        jumping_subtree_solutions: Dict[Partition, List[List[Partition]]],
    ):
        self.edges: List[Partition] = edges

        self.jumping_subtree_solutions: Dict[Partition, List[List[Partition]]] = (
            jumping_subtree_solutions
        )

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
        Get s-edges sorted by subset-aware depth with deterministic ties.

        Args:
            use_reference: If True, sort by reference tree depths; otherwise, use target tree depths.
            ascending: If True, sort from smallest to largest depth (subsets first).
            secondary_sort_key: Optional secondary key function.

        Tie-breaking (deterministic):
            - Depth (primary)
            - secondary_sort_key(p) if provided else 0
            - len(p)
            - tuple(sorted indices of p)
        """
        depth_map = self.reference_depths if use_reference else self.target_depths

        def sort_key(p: Partition):
            primary = depth_map.get(p, 0)
            secondary = secondary_sort_key(p) if secondary_sort_key else 0
            size = len(tuple(p))
            idxs = tuple(int(i) for i in p)
            return (primary, secondary, size, idxs)

        return sorted(self.edges, key=sort_key, reverse=not ascending)
