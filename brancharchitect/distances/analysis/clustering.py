"""
Clustering algorithms for distance matrices.

This module provides spectral clustering implementations with robust
NaN handling strategies for analyzing tree distance matrices.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering
from sklearn.impute import KNNImputer
from typing import Tuple, Literal


def perform_clustering(
    distance_matrix: NDArray[np.float64], n_clusters: int = 3
) -> NDArray[np.int64]:
    """
    Perform spectral clustering on the distance matrix.
    Handles NaN values by imputation or removal.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Symmetric distance matrix (will be sanitized)
    n_clusters : int
        Number of clusters to form

    Returns:
    --------
    NDArray[np.int64]
        Cluster labels for each sample

    Raises:
    -------
    ValueError
        If n_clusters is invalid or matrix has issues
    """
    from brancharchitect.distances.utils.matrix_utils import (
        sanitize_distance_matrix,
        robust_bandwidth,
        distance_to_similarity,
    )

    distance_matrix = sanitize_distance_matrix(distance_matrix)

    n_samples = distance_matrix.shape[0]
    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed number of samples ({n_samples})"
        )
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2 for spectral clustering")

    # For precomputed affinity, we need to convert the distance matrix to a similarity matrix
    sigma: float = robust_bandwidth(distance_matrix)
    similarity_matrix = distance_to_similarity(distance_matrix, sigma)

    clustering: SpectralClustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )

    cluster_labels: np.ndarray = clustering.fit_predict(similarity_matrix)
    return cluster_labels


def perform_clustering_robust(
    distance_matrix: NDArray[np.float64],
    n_clusters: int = 3,
    nan_strategy: Literal["median", "mean", "remove", "knn_impute"] = "median",
    min_valid_ratio: float = 0.5,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Enhanced clustering with robust NaN handling strategies.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Input distance matrix
    n_clusters : int
        Number of clusters
    nan_strategy : Literal["median", "mean", "remove", "knn_impute"]
        Strategy for handling NaN values:
        - "median": Replace with median distance
        - "mean": Replace with mean distance
        - "remove": Remove rows/columns with NaN values
        - "knn_impute": Use KNN imputation
    min_valid_ratio : float
        Minimum ratio of non-NaN values required per row/column

    Returns:
    --------
    cluster_labels : NDArray[np.int64]
        Cluster assignments
    valid_indices : NDArray[np.int64]
        Indices of trees included in clustering (useful when rows are removed)
    """
    print(f"Original matrix shape: {distance_matrix.shape}")
    print(f"NaN count: {np.sum(np.isnan(distance_matrix))}")

    original_indices: NDArray[np.int64] = np.arange(distance_matrix.shape[0])

    if not np.isnan(distance_matrix).any():
        print("No NaN values found - proceeding with standard clustering")
        return perform_clustering(distance_matrix, n_clusters), original_indices

    if nan_strategy == "remove":
        # Remove rows/columns with too many NaN values
        nan_counts_per_row: NDArray[np.int64] = np.sum(
            np.isnan(distance_matrix), axis=1
        )
        valid_ratio_per_row: NDArray[np.float64] = 1 - (
            nan_counts_per_row / distance_matrix.shape[1]
        )
        valid_rows: NDArray[np.bool_] = valid_ratio_per_row >= min_valid_ratio

        if np.sum(valid_rows) < n_clusters:
            print(
                f"Warning: Only {np.sum(valid_rows)} valid rows remain, but {n_clusters} clusters requested."
            )
            print("Falling back to median imputation strategy.")
            nan_strategy = "median"
        else:
            distance_matrix = distance_matrix[valid_rows][:, valid_rows]
            original_indices = original_indices[valid_rows]
            print(
                f"Removed {np.sum(~valid_rows)} rows/columns. New shape: {distance_matrix.shape}"
            )

    if nan_strategy in ["median", "mean", "knn_impute"]:
        distance_matrix = np.copy(distance_matrix)

        if nan_strategy == "median":
            non_nan_values: NDArray[np.float64] = distance_matrix[
                ~np.isnan(distance_matrix)
            ]
            if len(non_nan_values) == 0:
                raise ValueError("All values in distance matrix are NaN")
            fill_value: np.float64 = np.median(non_nan_values)
            distance_matrix[np.isnan(distance_matrix)] = fill_value
            print(f"Replaced NaN with median: {fill_value:.4f}")

        elif nan_strategy == "mean":
            non_nan_values = distance_matrix[~np.isnan(distance_matrix)]
            if len(non_nan_values) == 0:
                raise ValueError("All values in distance matrix are NaN")
            fill_value = np.mean(non_nan_values)
            distance_matrix[np.isnan(distance_matrix)] = fill_value
            print(f"Replaced NaN with mean: {fill_value:.4f}")

        elif nan_strategy == "knn_impute":
            imputer: KNNImputer = KNNImputer(
                n_neighbors=min(5, distance_matrix.shape[0] - 1)
            )
            imputed_matrix: np.ndarray = imputer.fit_transform(
                np.asarray(distance_matrix, dtype=np.float64)
            )
            distance_matrix = imputed_matrix
            print("Applied KNN imputation for NaN values")

    # Ensure matrix is symmetric and run clustering
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    cluster_labels: NDArray[np.int64] = perform_clustering(distance_matrix, n_clusters)

    return cluster_labels, original_indices


__all__ = [
    "perform_clustering",
    "perform_clustering_robust",
]
