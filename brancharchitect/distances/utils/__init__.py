"""
Distance matrix utilities.

Provides utility functions for preprocessing and transforming distance matrices,
including NaN handling, symmetrization, and similarity conversions.
"""

from brancharchitect.distances.utils.matrix_utils import (
    sanitize_distance_matrix,
    robust_bandwidth,
    distance_to_similarity,
)

__all__ = [
    "sanitize_distance_matrix",
    "robust_bandwidth",
    "distance_to_similarity",
]
