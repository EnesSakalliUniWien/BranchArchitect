import logging
import pandas as pd
from typing import List, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_tree_trajectories_scatter(
    pairwise_distances_list, method_names, window_size=10
):
    """
    Plot pairwise distances as trajectories with a clean design.

    Parameters
    ----------
    pairwise_distances_list : List[List[float]]
        Each element is a list of distances for a particular method.
    method_names : List[str]
        Names corresponding to each method.
    window_size : int
        Window size for moving average and standard deviation calculations.
    """
    if len(method_names) != len(pairwise_distances_list):
        raise ValueError(
            "method_names and pairwise_distances_list must have the same length."
        )

    n_methods = len(method_names)
    all_distances = np.concatenate(
        [np.array(d, dtype=float) for d in pairwise_distances_list]
    )
    global_min, global_max = np.min(all_distances), np.max(all_distances)

    # Color-blind-friendly palette
    cb_colors = {"blue": "#0072B2", "pink": "#CC79A7", "dark_gray": "#777777"}

    # Configure plot style
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "gray",
            "grid.color": "gray",
            "grid.alpha": 0.7,
            "font.size": 10,
            "axes.titlepad": 10,
        }
    )

    fig, axes = plt.subplots(n_methods, 1, figsize=(18, 5 * n_methods))
    axes = [axes] if n_methods == 1 else axes

    for idx, (distances, method_name) in enumerate(
        zip(pairwise_distances_list, method_names)
    ):
        distances = np.array(distances, dtype=float)
        x_pos = np.arange(len(distances))

        # Calculate total distance, moving average, and standard deviation
        total_distance = np.sum(distances)
        moving_avg = (
            savgol_filter(
                distances, window_length=min(window_size, len(distances)), polyorder=2
            )
            if len(distances) > 1
            else distances
        )
        window_std = pd.Series(distances).rolling(window_size, center=True).std()

        # Trajectory plot
        ax = axes[idx]
        ax.plot(
            x_pos,
            distances,
            color=cb_colors["blue"],
            alpha=0.85,
            linewidth=1.8,
            label="Distances",
        )
        ax.plot(
            x_pos,
            moving_avg,
            color=cb_colors["pink"],
            linewidth=2.0,
            label="Trend",
            alpha=0.9,
        )
        if len(window_std) == len(distances):
            ax.fill_between(
                x_pos,
                moving_avg - window_std,
                moving_avg + window_std,
                color=cb_colors["dark_gray"],
                alpha=0.2,
            )

        ax.set_ylim(
            global_min - 0.05 * (global_max - global_min),
            global_max + 0.05 * (global_max - global_min),
        )
        ax.set_title(
            f"{method_name} - Total Distance: {total_distance:.2f}", fontsize=16
        )
        ax.set_ylabel("Distance", fontsize=14)
        ax.set_xlabel("Tree Pair Index", fontsize=14)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Adjust x-ticks
        if len(x_pos) > 50:
            skip = max(len(x_pos) // 20, 1)
            ax.set_xticks(x_pos[::skip])
            ax.set_xticklabels(x_pos[::skip], rotation=45, fontsize=12)
        else:
            ax.set_xticks(x_pos)

    plt.tight_layout()
    plt.show()
    return fig

def plot_tree_subset_ratios(
    pairwise_distances_list: List[List[float]],
    method_names: List[str],
    split_distance_containers_list: List[Dict[str, List[float]]],
    epsilon=1e-9,
):
    """
    Plot ratio of subset distances to full distances using a consistent style.

    Parameters
    ----------
    pairwise_distances_list : List[List[float]]
        Each element is a list of full distances for a particular method.
    method_names : List[str]
        Names corresponding to each method.
    split_distance_containers_list : List[Dict[str, List[float]]]
        Each dict must contain "unique_splits1_distances", "unique_splits2_distances",
        and "common_splits_distances".
    epsilon : float
        Small constant to avoid division by zero.
    """

    n_methods = len(method_names)
    ratio_unique1_list = []
    ratio_unique2_list = []
    ratio_common_list = []

    for full_distances, split_container in zip(
        pairwise_distances_list, split_distance_containers_list
    ):
        ratio_unique1_list.append(
            np.array(split_container.get("unique_splits1_distances", []), dtype=float)
        )
        ratio_unique2_list.append(
            np.array(split_container.get("unique_splits2_distances", []), dtype=float)
        )
        ratio_common_list.append(
            np.array(split_container.get("common_splits_distances", []), dtype=float)
        )

    all_ratios = ratio_unique1_list + ratio_unique2_list + ratio_common_list
    global_min_ratio, global_max_ratio = np.min(np.concatenate(all_ratios)), np.max(
        np.concatenate(all_ratios)
    )

    # Vibrant theme parameters
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#DDDDDD",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#DDDDDD",
            "grid.alpha": 0.7,
            "font.size": 12,
            "font.family": "sans-serif",
        }
    )

    fig, axes = plt.subplots(n_methods, 1, figsize=(18, 5 * n_methods))
    axes = [axes] if n_methods == 1 else axes

    for idx, (method_name, ax) in enumerate(zip(method_names, axes)):
        x_pos = np.arange(len(pairwise_distances_list[idx]))
        unique1_total = np.sum(ratio_unique1_list[idx])
        unique2_total = np.sum(ratio_unique2_list[idx])
        common_total = np.sum(ratio_common_list[idx])

        ax.plot(
            x_pos,
            ratio_unique1_list[idx],
            color="#FFD700",
            linewidth=2.5,
            label=f"Unique1 Ratio (Sum: {unique1_total:.2f})",
            zorder=3,
        )
        ax.plot(
            x_pos,
            ratio_unique2_list[idx],
            color="#FF69B4",
            linewidth=2.5,
            label=f"Unique2 Ratio (Sum: {unique2_total:.2f})",
            zorder=3,
        )
        ax.plot(
            x_pos,
            ratio_common_list[idx],
            color="#32CD32",
            linewidth=2.5,
            label=f"Common Ratio (Sum: {common_total:.2f})",
            zorder=3,
        )

        ax.set_ylim(
            global_min_ratio - 0.05 * (global_max_ratio - global_min_ratio),
            global_max_ratio + 0.05 * (global_max_ratio - global_min_ratio),
        )
        ax.set_title(f"{method_name} - Ratio Plot", fontsize=16)
        ax.set_ylabel("Ratio", fontsize=14)
        ax.set_xlabel("Tree Pair Index", fontsize=14)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust x-ticks
        if len(x_pos) > 50:
            skip = max(len(x_pos) // 20, 1)
            ax.set_xticks(x_pos[::skip])
            ax.set_xticklabels(x_pos[::skip], rotation=45, fontsize=12)
        else:
            ax.set_xticks(x_pos)

    plt.tight_layout()
    plt.show()
    return fig


def plot_method_and_permutation_boxplots(
    method_names, total_distances, random_permutation_distances
):
    """
    Create a pastel-themed comparison plot using colors inspired by the provided image.
    """

    # Pastel theme parameters derived from the image
    bg_color = "#F5EEE6"
    text_color = "#5A4F4F"
    grid_color = "#C8BCB1"
    # A set of pastel colors reflecting the image's palette
    pastel_palette = ["#7FB5B5", "#D8A5B3", "#B39CD0", "#E1C16E"]

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.facecolor": bg_color,
            "axes.facecolor": bg_color,
            "axes.edgecolor": grid_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "grid.color": grid_color,
            "grid.alpha": 0.6,
            "font.size": 12,
        }
    )

    # Prepare data
    data = []
    for m_name, dist in zip(method_names, total_distances):
        data.append({"Method": m_name, "Distance": dist, "Type": "Method"})
    for dist in random_permutation_distances:
        data.append(
            {"Method": "Random Permutations", "Distance": dist, "Type": "Random"}
        )
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Assign colors
    color_map = {}
    for i, method in enumerate(method_names):
        color_map[method] = pastel_palette[i % len(pastel_palette)]
    # Neutral gray for random permutations
    color_map["Random Permutations"] = "#9E9E9E"

    # Plot boxplot for random permutations
    sns.boxplot(
        x="Method",
        y="Distance",
        data=df[df["Method"] == "Random Permutations"],
        order=["Random Permutations"],
        color=color_map["Random Permutations"],
        width=0.6,
        showfliers=False,
        boxprops={"alpha": 0.7, "zorder": 1},
        ax=ax,
    )

    # Add swarmplot for random permutations
    sns.swarmplot(
        x="Method",
        y="Distance",
        data=df[df["Method"] == "Random Permutations"],
        order=["Random Permutations"],
        color="#6E6E6E",
        size=4,
        alpha=0.5,
        ax=ax,
    )

    # Add method points
    for method in method_names:
        method_data = df[df["Method"] == method]
        # Glow effect
        ax.scatter(
            method_names.index(method),
            method_data["Distance"],
            s=150,
            color=color_map[method],
            alpha=0.3,
            zorder=3,
        )
        # Main point
        ax.scatter(
            method_names.index(method),
            method_data["Distance"],
            s=100,
            color=color_map[method],
            marker="D",
            edgecolor="white",
            linewidth=1,
            zorder=4,
            label=method,
        )

    ax.set_title(
        "Method Performance vs Random Permutations",
        pad=20,
        color=text_color,
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Methods", color=text_color, fontsize=14)
    ax.set_ylabel("Total Distance", color=text_color, fontsize=14)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add statistics box
    stats_text = (
        f"Random Permutations:\n"
        f"Mean: {np.mean(random_permutation_distances):.2f}\n"
        f"Std: {np.std(random_permutation_distances):.2f}\n"
        f"Best Method: {method_names[np.argmin(total_distances)]}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=12,
        color=text_color,
        bbox=dict(
            boxstyle="round", facecolor="#EFE8DE", edgecolor="#BEB5A9", alpha=0.8
        ),
    )

    plt.xticks(rotation=45, ha="right", color=text_color)
    plt.tight_layout()
    return fig