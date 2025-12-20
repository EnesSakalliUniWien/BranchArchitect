"""
Distance plotting submodule.

Provides visualization functions for distance matrices and embeddings,
including heatmaps, scatter plots, and interactive 3D visualizations.
"""

from brancharchitect.distances.plotting.matrix_plots import (
    plot_distance_matrix,
    plot_consecutive_distances,
    plot_component_distance_matrix,
    plot_component_consecutive_distances,
)
from brancharchitect.distances.plotting.scatter_plots import (
    create_3d_scatter_plot,
    plot_component_umap_3d,
    plot_component_umap_3d_with_graph,
)

__all__ = [
    "plot_distance_matrix",
    "plot_consecutive_distances",
    "plot_component_distance_matrix",
    "plot_component_consecutive_distances",
    "create_3d_scatter_plot",
    "plot_component_umap_3d",
    "plot_component_umap_3d_with_graph",
]
