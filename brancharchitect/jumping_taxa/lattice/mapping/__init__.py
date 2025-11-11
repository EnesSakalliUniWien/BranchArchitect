"""Lattice mapping utilities."""

from .iterative_pivot_mappings import (
    map_iterative_pivot_edges_to_original,
)
from .minimum_cover_mappings import (
    find_best_overlapping_partition,
    map_solution_elements_to_minimum_covers,
    map_solution_elements_to_minimal_frontiers,
)

# Backward-compatible aliases (older naming: "s-edges" or "transient")
map_iterative_sedges_to_original = map_iterative_pivot_edges_to_original
map_transient_sedges_to_original = map_iterative_pivot_edges_to_original

__all__ = [
    "map_iterative_pivot_edges_to_original",
    "map_iterative_sedges_to_original",
    "map_transient_sedges_to_original",
    "find_best_overlapping_partition",
    "map_solution_elements_to_minimum_covers",
    "map_solution_elements_to_minimal_frontiers",
]
