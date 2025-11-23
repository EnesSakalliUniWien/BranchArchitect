"""
Dimensionality reduction algorithms for distance matrices.

This module provides UMAP-based dimensionality reduction with
robust NaN handling for visualizing high-dimensional tree spaces.
"""

import numpy as np
import umap
from numpy.typing import NDArray
from typing import Union, Tuple


def perform_umap(
    distance_matrix: NDArray[np.float64],
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    return_model: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], umap.UMAP]]:
    """
    Perform UMAP dimensionality reduction on the distance matrix.
    Handles NaN values by imputation.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Input distance matrix
    n_components : int
        Number of dimensions for the embedding
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance for UMAP
    return_model : bool
        If True, returns tuple of (embedding, model), otherwise just embedding

    Returns:
    --------
    Union[NDArray[np.float64], Tuple[NDArray[np.float64], umap.UMAP]]
        Either just the embedding or tuple of (embedding, fitted_model)

    Examples:
    ---------
    >>> # Basic usage
    >>> embedding = perform_umap(distance_matrix)
    >>>
    >>> # Get model for visualization
    >>> embedding, model = perform_umap(distance_matrix, return_model=True)
    """
    # Check for NaN values and handle them
    if np.isnan(distance_matrix).any():
        print(
            f"Warning: Found {np.sum(np.isnan(distance_matrix))} NaN values in distance matrix for UMAP. Applying imputation..."
        )

        # Create a copy to avoid modifying the original matrix
        distance_matrix = np.copy(distance_matrix)

        # Calculate median of non-NaN values
        non_nan_values: NDArray[np.float64] = distance_matrix[
            ~np.isnan(distance_matrix)
        ]
        if len(non_nan_values) == 0:
            raise ValueError("All values in distance matrix are NaN")

        median_distance: np.float64 = np.median(non_nan_values)

        # Replace NaN values with median
        distance_matrix[np.isnan(distance_matrix)] = median_distance
        print(f"Replaced NaN values with median distance: {median_distance:.4f}")

    # Ensure matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=42,
        spread=1.0,
        densmap=True,
    )
    embedding: NDArray[np.float64] = umap_model.fit_transform(distance_matrix)

    if return_model:
        return embedding, umap_model
    return embedding


__all__ = [
    "perform_umap",
]
