"""
Matrix preprocessing and transformation utilities.

Helper functions for sanitizing distance matrices, computing bandwidths,
and converting between distance and similarity representations.
"""

import numpy as np
from numpy.typing import NDArray


def sanitize_distance_matrix(
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Ensure symmetry, zero diagonal, and impute NaNs with median of observed values.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Input distance matrix (may be asymmetric or contain NaNs)

    Returns:
    --------
    NDArray[np.float64]
        Sanitized symmetric distance matrix with zero diagonal

    Raises:
    -------
    ValueError
        If all values in distance matrix are NaN
    """
    distance_array: NDArray[np.float64] = np.asarray(distance_matrix, dtype=float)
    if np.isnan(distance_array).any():
        print(
            f"Warning: Found {np.sum(np.isnan(distance_array))} NaN values in distance matrix. Applying imputation..."
        )
        non_nan_values: NDArray[np.float64] = distance_array[~np.isnan(distance_array)]
        if len(non_nan_values) == 0:
            raise ValueError("All values in distance matrix are NaN")
        median_distance: np.float64 = np.median(non_nan_values)
        distance_array = np.where(
            np.isnan(distance_array), median_distance, distance_array
        )
        print(f"Replaced NaN values with median distance: {median_distance:.4f}")

    distance_array = (distance_array + distance_array.T) / 2
    np.fill_diagonal(distance_array, 0.0)
    return distance_array


def robust_bandwidth(distance_matrix: NDArray[np.float64]) -> float:
    """
    Compute a robust bandwidth (sigma) from off-diagonal distances using median/IQR.
    Avoids diagonal zeros shrinking the scale.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix (should be sanitized first)

    Returns:
    --------
    float
        Robust bandwidth estimate (minimum 1e-8)
    """
    indices = np.triu_indices_from(distance_matrix, k=1)
    upper = distance_matrix[indices]
    upper = upper[~np.isnan(upper)]
    if upper.size == 0:
        return 1.0
    median = float(np.median(upper))
    iqr = float(np.percentile(upper, 75) - np.percentile(upper, 25))
    sigma_candidates = [median, iqr / 1.349 if iqr > 0 else 0.0, float(np.std(upper))]
    sigma = (
        max(c for c in sigma_candidates if c > 0)
        if any(c > 0 for c in sigma_candidates)
        else median
    )
    return max(sigma, 1e-8)


def distance_to_similarity(
    distance_matrix: NDArray[np.float64], sigma: float
) -> NDArray[np.float64]:
    """
    Convert distances to a positive similarity matrix using an RBF kernel.

    Uses the formula: similarity = exp(-distance / sigma)

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix
    sigma : float
        Bandwidth parameter for RBF kernel (must be positive)

    Returns:
    --------
    NDArray[np.float64]
        Similarity matrix with diagonal set to 1.0

    Raises:
    -------
    ValueError
        If sigma is not positive
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    similarity = np.exp(-distance_matrix / sigma)
    np.fill_diagonal(similarity, 1.0)
    return similarity


__all__ = [
    "sanitize_distance_matrix",
    "robust_bandwidth",
    "distance_to_similarity",
]
