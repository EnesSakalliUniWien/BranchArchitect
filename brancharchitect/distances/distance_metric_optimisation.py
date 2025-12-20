# Cell 1: Imports and Initial Setup
import numpy as np
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Any
from numpy.typing import NDArray
from brancharchitect.distances.analysis.clustering import perform_clustering


def objective_function(
    weights: Tuple[float, float, float],
    results: List[Tuple[int, int, List[List[Any]], Any, List[Any], List[Any]]],
    max_trees: int,
    max_component_sum: int,
    max_num_solutions: int,
    max_path_length: int,
) -> float:
    """Objective function to maximize clustering quality"""
    w1: float
    w2: float
    w3: float
    w1, w2, w3 = weights

    # Normalize weights
    total: float = w1 + w2 + w3
    if total == 0:
        return 1  # Return bad score for zero weights

    w1_norm: float = w1 / total
    w2_norm: float = w2 / total
    w3_norm: float = w3 / total

    weighted_matrix: NDArray[np.float64] = compute_weighted_distance_matrix(
        results,
        max_trees,
        w1_norm,
        w2_norm,
        w3_norm,
        max_component_sum,
        max_num_solutions,
        max_path_length,
    )

    score: float = evaluate_clustering_quality(weighted_matrix, n_clusters=3)

    # Return negative score because minimize() minimizes
    return -score


def compute_weighted_distance_matrix(
    results: List[Tuple[int, int, List[List[Any]], Any, List[Any], List[Any]]],
    max_trees: int,
    w1: float,
    w2: float,
    w3: float,
    max_component_sum: int,
    max_num_solutions: int,
    max_path_length: int,
) -> NDArray[np.float64]:
    """Compute weighted distance matrix with given weights"""
    weighted_matrix: NDArray[np.float64] = np.zeros((max_trees, max_trees), dtype=float)

    for i, j, components, _, path_i, path_j in results:
        component_sum: int = sum(len(c) for c in components)
        num_solutions: int = len(components)
        path_lengths: int = len(path_i) + len(path_j)

        norm_component_sum: float = (
            component_sum / max_component_sum if max_component_sum else 0
        )
        norm_num_solutions: float = (
            num_solutions / max_num_solutions if max_num_solutions else 0
        )
        norm_path_length: float = (
            path_lengths / max_path_length if max_path_length else 0
        )

        dist: float = (
            w1 * norm_component_sum + w2 * norm_num_solutions + w3 * norm_path_length
        )
        weighted_matrix[i, j] = dist
        weighted_matrix[j, i] = dist

    return weighted_matrix


def evaluate_clustering_quality(
    distance_matrix: NDArray[np.float64], n_clusters: int = 3
) -> float:
    """Evaluate clustering quality using silhouette score"""
    if np.all(distance_matrix == 0):
        return -1

    cluster_labels: NDArray[np.int64] = perform_clustering(
        distance_matrix, n_clusters=n_clusters
    )

    # Convert distance to similarity for silhouette score
    max_dist: np.float64 = np.max(distance_matrix)
    if max_dist > 0:
        similarity_matrix: NDArray[np.float64] = 1 - (distance_matrix / max_dist)
    else:
        return -1

    try:
        score_result = silhouette_score(
            similarity_matrix, cluster_labels, metric="precomputed"
        )
        score: float = float(score_result)
        return score
    except Exception:
        return -1
