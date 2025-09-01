"""
Classical interpolation algorithms.
"""

from .classical_interpolation import (
    interpolate_tree,
    interpolate_adjacent_tree_pairs,
)

__all__ = [
    "interpolate_tree", 
    "interpolate_adjacent_tree_pairs",
]