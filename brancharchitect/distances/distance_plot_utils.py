# Cell 1: Imports and Initial Setup
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import plotly.graph_objs as go
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from numpy.typing import NDArray
from typing import List, Optional, Union, Tuple, Literal, cast
from scipy.sparse import coo_matrix


def plot_consecutive_distances(
    df: pd.DataFrame, distance_metric: str, ax: Axes, title: Optional[str] = None
) -> None:
    """
    Plot distances between consecutive trees as a line plot.
    """
    df_consecutive: pd.DataFrame = df[df["Tree2"] == df["Tree1"] + 1]
    df_consecutive = df_consecutive.sort_values(by="Tree1")  # type: ignore[assignment]

    ax.plot(
        df_consecutive["Tree1"].to_numpy(),
        df_consecutive[distance_metric].to_numpy(dtype=float),
        marker="o",
        linestyle="-",
    )
    if not title:
        title = f"Consecutive Distances ({distance_metric})"
    ax.set_title(title)
    ax.set_xlabel("Tree Index")
    ax.set_ylabel(f"{distance_metric} Distance")
    ax.grid(True)


def plot_distance_matrix(
    distance_matrix: Union[NDArray[np.float64], pd.DataFrame],
    ax: Axes,
    title: str = "Distance Matrix",
) -> None:
    """
    Plot the distance matrix as a heatmap.
    """
    sns.heatmap(np.asarray(distance_matrix), cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tree Index")
    ax.set_ylabel("Tree Index")


def perform_clustering(
    distance_matrix: NDArray[np.float64], n_clusters: int = 3
) -> NDArray[np.int64]:
    """
    Perform spectral clustering on the distance matrix.
    Handles NaN values by imputation or removal.
    """
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


def scan_silhouette_scores(
    distance_matrix: NDArray[np.float64],
    k_values: List[int],
    random_state: int = 42,
) -> List[Tuple[int, float]]:
    """
    Compute silhouette scores over a range of cluster counts using spectral clustering.
    Returns list of (k, score) where invalid ks yield np.nan.
    """
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
    """
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
    """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_distance_matrix(
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ensure symmetry, zero diagonal, and impute NaNs with median of observed values."""
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
    """
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
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
    """Convert distances to a positive similarity matrix using an RBF kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    similarity = np.exp(-distance_matrix / sigma)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _total_intra_cluster_distance(
    distance_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    """Sum of intra-cluster distances (upper triangle) for the given labels."""
    total = 0.0
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        total += float(sub[np.triu_indices_from(sub, k=1)].sum())
    return total


def create_3d_scatter_plot(
    embedding: NDArray[np.float64],
    tree_indices: List[int],
    labels: NDArray[np.int64],
    title: str,
    method_name: str,
) -> None:
    """
    Create a 3D scatter plot for the embedding.
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


def plot_component_distance_matrix(
    distance_matrix: Union[NDArray[np.float64], pd.DataFrame],
    title: str = "Component Distance Matrix",
) -> None:
    """
    Plot the component distance matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Tree Index")
    plt.ylabel("Tree Index")
    plt.show()


def plot_component_consecutive_distances(
    df_distances: pd.DataFrame, ax: Optional[Axes] = None, title: Optional[str] = None
) -> None:
    """
    Plot distances between consecutive trees as a line plot.
    """
    created_axes: bool = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        created_axes = True
    df_consecutive: pd.DataFrame = df_distances[
        df_distances["Tree2"] == df_distances["Tree1"] + 1
    ]
    df_consecutive = df_consecutive.sort_values(by="Tree1")
    ax.plot(
        df_consecutive["Tree1"],
        df_consecutive["ComponentDistance"],
        marker="o",
        linestyle="-",
    )
    if not title:
        title = "Consecutive Component Distances"
    ax.set_title(title)
    ax.set_xlabel("Tree Index")
    ax.set_ylabel("Component Distance")
    ax.grid(True)
    if created_axes:
        plt.show()


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
        The UMAP embedding coordinates
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
    distance_matrix : NDArray[np.floating]
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
    cluster_labels : NDArray[np.integer]
        Cluster assignments
    valid_indices : NDArray[np.integer]
        Indices of trees included in clustering (useful when rows are removed)
    """
    from sklearn.impute import KNNImputer

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
    """
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
