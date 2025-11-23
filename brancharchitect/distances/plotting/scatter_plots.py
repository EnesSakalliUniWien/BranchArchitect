"""
3D scatter plot visualizations.

Provides interactive 3D scatter plots using Plotly for visualizing
embeddings and UMAP results with optional graph structures.
"""

import numpy as np
import umap
import plotly.graph_objs as go
from numpy.typing import NDArray
from typing import List, Optional, cast, Tuple


def create_3d_scatter_plot(
    embedding: NDArray[np.float64],
    tree_indices: List[int],
    labels: NDArray[np.int64],
    title: str,
    method_name: str,
) -> None:
    """
    Create a 3D scatter plot for the embedding.

    Parameters:
    -----------
    embedding : NDArray[np.float64]
        3D embedding coordinates (n_samples, 3)
    tree_indices : List[int]
        Tree indices for hover labels
    labels : NDArray[np.int64]
        Cluster labels for coloring
    title : str
        Plot title
    method_name : str
        Name of the embedding method (for axis labels)

    Examples:
    ---------
    >>> create_3d_scatter_plot(
    ...     embedding=umap_embedding,
    ...     tree_indices=list(range(n_trees)),
    ...     labels=cluster_labels,
    ...     title="UMAP Embedding",
    ...     method_name="UMAP"
    ... )
    """
    # Use Plotly for interactive 3D plotting
    fig: go.Figure = go.Figure(
        data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=labels,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Cluster"),
                ),
                text=[f"Tree Index: {idx}" for idx in tree_indices],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{method_name} Dimension 1",
            yaxis_title=f"{method_name} Dimension 2",
            zaxis_title=f"{method_name} Dimension 3",
        ),
        width=800,
        height=600,
    )

    fig.show()


def plot_component_umap_3d(
    embedding: NDArray[np.float64],
    cluster_labels: NDArray[np.int64],
    title: str = "UMAP Embedding (Component Distance)",
    umap_model: Optional[umap.UMAP] = None,
    show_graph: bool = False,
    graph_opacity: float = 0.1,
) -> None:
    """
    Create a 3D scatter plot for the UMAP embedding of component distances.

    Parameters:
    -----------
    embedding : NDArray[np.float64]
        The UMAP embedding coordinates (n_samples, 3)
    cluster_labels : NDArray[np.int64]
        Cluster labels for coloring points
    title : str
        Title for the plot
    umap_model : Optional[umap.UMAP]
        The fitted UMAP model (required if show_graph=True)
    show_graph : bool
        Whether to show the UMAP k-nearest neighbor graph structure
    graph_opacity : float
        Opacity of the graph edges (0.0 to 1.0, default 0.1)

    Examples:
    ---------
    >>> # Basic usage
    >>> plot_component_umap_3d(embedding, cluster_labels)
    >>>
    >>> # With graph structure
    >>> embedding, model = perform_umap(dist_matrix, return_model=True)
    >>> plot_component_umap_3d(
    ...     embedding, cluster_labels,
    ...     umap_model=model,
    ...     show_graph=True
    ... )
    """
    traces = []

    # Add the main scatter plot
    traces.append(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=cluster_labels,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Cluster"),
            ),
            text=[f"Tree {i + 1}" for i in range(embedding.shape[0])],
            hoverinfo="text",
            name="Data Points",
        )
    )

    # Add graph edges if requested and UMAP model is provided
    if show_graph and umap_model is not None:
        # Extract the k-nearest neighbor graph from UMAP
        if hasattr(umap_model, "graph_"):
            graph = umap_model.graph_
            # Convert sparse matrix to coordinate format for easier access
            coo_graph = graph.tocoo()

            # Prepare edge coordinates for plotting
            edge_x = []
            edge_y = []
            edge_z = []

            for i in range(len(coo_graph.row)):
                source_idx = coo_graph.row[i]
                target_idx = coo_graph.col[i]

                # Add edge coordinates
                edge_x.extend(
                    [embedding[source_idx, 0], embedding[target_idx, 0], None]
                )
                edge_y.extend(
                    [embedding[source_idx, 1], embedding[target_idx, 1], None]
                )
                edge_z.extend(
                    [embedding[source_idx, 2], embedding[target_idx, 2], None]
                )

            # Add graph edges as a separate trace
            traces.append(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color=f"rgba(125,125,125,{graph_opacity})", width=1),
                    hoverinfo="none",
                    name="UMAP Graph",
                    showlegend=True,
                )
            )

    fig: go.Figure = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
        ),
        width=800,
        height=600,
    )
    fig.show()


def plot_component_umap_3d_with_graph(
    distance_matrix: NDArray[np.float64],
    cluster_labels: NDArray[np.int64],
    title: str = "UMAP Embedding with Graph Structure",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    show_graph: bool = True,
) -> None:
    """
    Convenience function that performs UMAP and plots the result with graph structure.

    Parameters:
    -----------
    distance_matrix : NDArray[np.float64]
        Distance matrix for UMAP computation
    cluster_labels : NDArray[np.int64]
        Cluster labels for coloring points
    title : str
        Title for the plot
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance for UMAP
    show_graph : bool
        Whether to show the UMAP k-nearest neighbor graph structure

    Examples:
    ---------
    >>> plot_component_umap_3d_with_graph(
    ...     distance_matrix=dist_matrix,
    ...     cluster_labels=labels,
    ...     title="Tree Space UMAP",
    ...     show_graph=True
    ... )
    """
    from brancharchitect.distances.analysis.dimensionality import perform_umap

    # Perform UMAP and get both embedding and model
    result = perform_umap(
        distance_matrix,
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        return_model=True,
    )

    # Type guard: when return_model=True, result is always a tuple
    embedding, umap_model = cast(Tuple[NDArray[np.float64], umap.UMAP], result)

    # Plot with graph structure
    plot_component_umap_3d(
        embedding=embedding,
        cluster_labels=cluster_labels,
        title=title,
        umap_model=umap_model,
        show_graph=show_graph,
    )


__all__ = [
    "create_3d_scatter_plot",
    "plot_component_umap_3d",
    "plot_component_umap_3d_with_graph",
]
