"""
Matrix visualization functions.

Provides heatmaps and line plots for visualizing distance matrices
and consecutive tree distances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import seaborn as sns
from numpy.typing import NDArray
from typing import Optional, Union, cast


def plot_distance_matrix(
    distance_matrix: Union[NDArray[np.float64], pd.DataFrame],
    ax: Axes,
    title: str = "Distance Matrix",
) -> None:
    """
    Plot the distance matrix as a heatmap.

    Parameters:
    -----------
    distance_matrix : Union[NDArray[np.float64], pd.DataFrame]
        Distance matrix to visualize
    ax : Axes
        Matplotlib axes to plot on
    title : str
        Plot title

    Examples:
    ---------
    >>> fig, ax = plt.subplots()
    >>> plot_distance_matrix(dist_matrix, ax, title="Tree Distances")
    >>> plt.show()
    """
    sns.heatmap(np.asarray(distance_matrix), cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tree Index")
    ax.set_ylabel("Tree Index")


def plot_consecutive_distances(
    df: pd.DataFrame, distance_metric: str, ax: Axes, title: Optional[str] = None
) -> None:
    """
    Plot distances between consecutive trees as a line plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Tree1, Tree2, and distance_metric
    distance_metric : str
        Name of the distance metric column to plot
    ax : Axes
        Matplotlib axes to plot on
    title : Optional[str]
        Plot title (auto-generated if None)

    Examples:
    ---------
    >>> fig, ax = plt.subplots()
    >>> plot_consecutive_distances(df, "RF_Distance", ax)
    >>> plt.show()
    """
    df_consecutive = cast(pd.DataFrame, df[df["Tree2"] == df["Tree1"] + 1])
    df_consecutive = df_consecutive.sort_values(by="Tree1")

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


def plot_component_distance_matrix(
    distance_matrix: Union[NDArray[np.float64], pd.DataFrame],
    title: str = "Component Distance Matrix",
) -> None:
    """
    Plot the component distance matrix as a heatmap.

    Creates a new figure and displays it immediately.

    Parameters:
    -----------
    distance_matrix : Union[NDArray[np.float64], pd.DataFrame]
        Distance matrix to visualize
    title : str
        Plot title

    Examples:
    ---------
    >>> plot_component_distance_matrix(component_dist_matrix)
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

    Parameters:
    -----------
    df_distances : pd.DataFrame
        DataFrame with columns: Tree1, Tree2, ComponentDistance
    ax : Optional[Axes]
        Matplotlib axes to plot on (creates new figure if None)
    title : Optional[str]
        Plot title (auto-generated if None)

    Examples:
    ---------
    >>> plot_component_consecutive_distances(df)
    >>> # Or with existing axes
    >>> fig, ax = plt.subplots()
    >>> plot_component_consecutive_distances(df, ax=ax)
    """
    created_axes: bool = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        created_axes = True
    df_consecutive = cast(
        pd.DataFrame, df_distances[df_distances["Tree2"] == df_distances["Tree1"] + 1]
    )
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


__all__ = [
    "plot_distance_matrix",
    "plot_consecutive_distances",
    "plot_component_distance_matrix",
    "plot_component_consecutive_distances",
]
