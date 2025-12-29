"""
Edge Ordering
=============

Functions for ordering pivot edges in the lattice algorithm.
Ensures proper processing order (subsets before supersets).
"""

from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    topological_sort_edges,
    sort_pivot_edges_by_subset_hierarchy,
)

__all__ = [
    "topological_sort_edges",
    "sort_pivot_edges_by_subset_hierarchy",
]
