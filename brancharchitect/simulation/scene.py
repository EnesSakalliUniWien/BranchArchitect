
import numpy as np

def generate_scene(rng, n_points, n_centroids, sigma_target, symmetric_cluster_ids: list[int] = None):
    """Generate initial centroids, clusters, and motion targets."""
    centroids = rng.uniform(-1.3, 1.3, size=(n_centroids, 3))
    clusters = rng.integers(0, n_centroids, size=n_points)

    if symmetric_cluster_ids is None:
        symmetric_cluster_ids = []

    # Generate targets with flexible symmetry
    centroid_targets = centroids + rng.normal(0, sigma_target, size=(n_centroids, 3))
    individual_targets = centroids[clusters] + rng.normal(
        0, sigma_target, size=(n_points, 3)
    )
    is_symmetric_mask = np.isin(clusters, symmetric_cluster_ids)
    targets = np.where(
        is_symmetric_mask[:, np.newaxis], centroid_targets[clusters], individual_targets
    )

    return centroids, clusters, targets, centroid_targets
