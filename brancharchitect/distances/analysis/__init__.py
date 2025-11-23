"""
Distance analysis submodule.

Provides clustering, dimensionality reduction, and quality metrics
for analyzing distance matrices in tree space.
"""

from brancharchitect.distances.analysis.clustering import (
    perform_clustering,
    perform_clustering_robust,
)
from brancharchitect.distances.analysis.dimensionality import perform_umap
from brancharchitect.distances.analysis.metrics import (
    scan_silhouette_scores,
    compute_eigengap_spectrum,
    elbow_intra_cluster_distances,
)

__all__ = [
    "perform_clustering",
    "perform_clustering_robust",
    "perform_umap",
    "scan_silhouette_scores",
    "compute_eigengap_spectrum",
    "elbow_intra_cluster_distances",
]
