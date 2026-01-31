"""
Tree processing services.

This package handles tree parsing, interpolation, and response building.
"""

from webapp.services.trees.processing import handle_tree_content
from webapp.services.trees.movie_data import MovieData

__all__ = [
    "handle_tree_content",
    "MovieData",
]
