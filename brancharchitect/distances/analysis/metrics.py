"""
Clustering quality metrics for distance matrices.

Provides silhouette scores, eigengap analysis, and elbow methods
for determining optimal cluster numbers.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from typing import List, Tuple


def scan_silhouette_scores(
    distance_matrix: NDArray[np.float64],
    k_values: List[int],
    random_state: int = 42,
) -> List[Tuple[int, float]]:
    """
    Compute silhouette scores over a range of cluster counts using spectral clustering.
    Returns list of (k, score) where invalid ks yield np.nan.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Precomputed distance matrix
    k_values : List[int]
        List of cluster numbers to evaluate
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    List[Tuple[int, float]]
        List of (k, silhouette_score) tuples

    Examples:
    ---------
    >>> scores = scan_silhouette_scores(dist_matrix, k_values=[2, 3, 4, 5])
    >>> best_k = max(scores, key=lambda x: x[1])[0]
    """
    from brancharchitect.distances.utils.matrix_utils import (
        sanitize_distance_matrix,
        robust_bandwidth,
        distance_to_similarity,
    )

    distance_matrix = sanitize_distance_matrix(distance_matrix)
    sigma = robust_bandwidth(distance_matrix)
    similarity_matrix = distance_to_similarity(distance_matrix, sigma)

    scores: List[Tuple[int, float]] = []
    n_samples = distance_matrix.shape[0]

    for k in k_values:
        if k < 2 or k > n_samples:
            scores.append((k, np.nan))
            continue
        try:
            labels = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state,
            ).fit_predict(similarity_matrix)
            score = float(
                silhouette_score(distance_matrix, labels, metric="precomputed")
            )
        except Exception:
            score = np.nan
        scores.append((k, score))
    return scores


def compute_eigengap_spectrum(
    distance_matrix: NDArray[np.float64],
    k_max: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized Laplacian eigenvalues and eigengaps up to k_max components.
    Returns (eigenvalues_sorted, eigengaps) where eigengaps[i] = lambda_{i+1} - lambda_i.

    The eigengap heuristic suggests the optimal number of clusters corresponds to
    the largest eigengap.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix
    k_max : int
        Maximum number of eigenvalues to compute

    Returns:
    --------
    eigenvalues_sorted : np.ndarray
        Sorted eigenvalues of normalized Laplacian
    eigengaps : np.ndarray
        Differences between consecutive eigenvalues

    Examples:
    ---------
    >>> eigvals, eigengaps = compute_eigengap_spectrum(dist_matrix)
    >>> optimal_k = np.argmax(eigengaps) + 1
    """
    from brancharchitect.distances.utils.matrix_utils import (
        sanitize_distance_matrix,
        robust_bandwidth,
        distance_to_similarity,
    )

    distance_matrix = sanitize_distance_matrix(distance_matrix)
    sigma = robust_bandwidth(distance_matrix)
    similarity_matrix = distance_to_similarity(distance_matrix, sigma)

    # Normalized Laplacian L = I - D^{-1/2} S D^{-1/2}
    degrees = similarity_matrix.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.diag(1.0 / np.sqrt(np.clip(degrees, 1e-12, None)))
    laplacian = (
        np.eye(similarity_matrix.shape[0])
        - inv_sqrt_deg @ similarity_matrix @ inv_sqrt_deg
    )

    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals_sorted = np.sort(eigvals)
    k_max = min(k_max, len(eigvals_sorted))
    eigvals_sorted = eigvals_sorted[:k_max]
    eigengaps = np.diff(eigvals_sorted)
    return eigvals_sorted, eigengaps


def elbow_intra_cluster_distances(
    distance_matrix: NDArray[np.float64],
    k_values: List[int],
    random_state: int = 42,
) -> List[Tuple[int, float]]:
    """
    Compute total intra-cluster distance for a range of k (useful for elbow plots).

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix
    k_values : List[int]
        List of cluster numbers to evaluate
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    List[Tuple[int, float]]
        List of (k, total_intra_cluster_distance) tuples

    Examples:
    ---------
    >>> elbow_data = elbow_intra_cluster_distances(dist_matrix, range(2, 11))
    >>> # Plot to find elbow point visually
    """
    from brancharchitect.distances.utils.matrix_utils import (
        sanitize_distance_matrix,
        robust_bandwidth,
        distance_to_similarity,
    )

    distance_matrix = sanitize_distance_matrix(distance_matrix)
    sigma = robust_bandwidth(distance_matrix)
    similarity_matrix = distance_to_similarity(distance_matrix, sigma)
    n_samples = distance_matrix.shape[0]

    totals: List[Tuple[int, float]] = []
    for k in k_values:
        if k < 1 or k > n_samples:
            totals.append((k, np.nan))
            continue
        labels = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        ).fit_predict(similarity_matrix)
        totals.append((k, _total_intra_cluster_distance(distance_matrix, labels)))
    return totals


def _total_intra_cluster_distance(
    distance_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    """
    Sum of intra-cluster distances (upper triangle) for the given labels.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix
    labels : NDArray[np.int64]
        Cluster labels for each sample

    Returns:
    --------
    float
        Total intra-cluster distance
    """
    total = 0.0
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        total += float(sub[np.triu_indices_from(sub, k=1)].sum())
    return total


__all__ = [
    "scan_silhouette_scores",
    "compute_eigengap_spectrum",
    "elbow_intra_cluster_distances",
]
