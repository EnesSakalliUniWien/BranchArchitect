"""
Tree interpolation module for phylogenetic tree analysis.

This module provides algorithms for creating smooth animations between
phylogenetic trees using lattice-based interpolation methods.

The module is organized into several submodules:
- core: Core tree calculation algorithms
- types: Data classes and type definitions
- ordering: Tree ordering and reordering functions
- helpers: Helper and processing functions
- refactored: Main public API functions

Main functions:
- interpolate_tree: Basic interpolation between two trees
- interpolate_adjacent_tree_pairs: Interpolate sequential tree pairs
"""

from brancharchitect.tree_interpolation.classical_interpolation import (
    interpolate_adjacent_tree_pairs,
    interpolate_tree,
)

__all__: list[str] = [
    "interpolate_tree",
    "interpolate_adjacent_tree_pairs",
]
