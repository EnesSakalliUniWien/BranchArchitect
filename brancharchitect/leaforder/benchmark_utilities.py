import logging
import cProfile
import pstats
import pandas as pd
import plotly.express as px
from typing import List, Set, Tuple, Dict
from brancharchitect.io import read_newick
from brancharchitect.tree import Node
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from brancharchitect.partition_set import Partition

# OLD classic approach
from brancharchitect.leaforder.tree_order_optimisation_local import (
    smooth_order_of_trees_classic,  # local iterative approach
    smooth_order_unique_sedge,
    optimize_s_edge_splits,  # local s-edge flips
    optimize_unique_splits,  # local unique splits flips
)

from brancharchitect.leaforder.circular_distances import (
    circular_distance_for_node_subset,
)


# GLOBAL approach
from brancharchitect.leaforder.tree_order_optimisation_global import (
    collect_distances_for_trajectory,
    generate_permutations,
    find_minimal_distance_permutation,
)

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


def profile_and_visualize(
    filepath,
    optimize_both_sides=True,
    backward=False,
    iterations=20,
    midpoint_rooted=True,
    min_time_percent=1.0,  # Only show functions taking >1% of total time
    focus_paths=None,  # Optional list of path substrings to focus on
):
    """
    Enhanced profiling and visualization of multiple ordering methods.

    Args:
        filepath: Path to input Newick file
        optimize_both_sides: Whether to optimize in both directions
        backward: Whether to do backward pass
        iterations: Number of optimization iterations
        midpoint_rooted: Whether to use midpoint rooting
        min_time_percent: Minimum percentage of total time to include function
        focus_paths: List of path substrings to focus visualization on
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    profiler = cProfile.Profile()

    try:
        logging.info("Starting profiling...")
        profiler.enable()

        benchmark_comparison(
            file_path=filepath,
            optimize_both_sides=optimize_both_sides,
            backward=backward,
            iterations=iterations,
            midpoint_root=midpoint_rooted,
        )

        profiler.disable()
        logging.info("Profiling completed successfully.")

    except Exception as e:
        profiler.disable()
        logging.error(f"An error occurred during profiling: {e}")
        raise e

    # Convert profile stats to DataFrame
    stats = pstats.Stats(profiler)
    total_time = stats.total_tt

    profile_data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        # Calculate additional metrics
        time_percent = (tt / total_time) * 100
        time_per_call = tt / cc if cc > 0 else 0

        # Skip if below minimum time threshold
        if time_percent < min_time_percent:
            continue

        # Filter by focus paths if specified
        if focus_paths:
            if not any(path in func[0] for path in focus_paths):
                continue

        # Shorten module name
        module_parts = func[0].split("/")
        short_module = module_parts[-1] if module_parts else func[0]

        profile_data.append(
            {
                "Function": f"{func[2]} ({short_module}:{func[1]})",
                "Module": short_module,
                "Line": func[1],
                "Name": func[2],
                "Calls": cc,
                "Total Time (s)": tt,
                "Time/Call (ms)": time_per_call * 1000,
                "Cumulative (s)": ct,
                "Time %": time_percent,
                "Callers": len(callers),
            }
        )

    df = pd.DataFrame(profile_data)
    df = df.sort_values("Total Time (s)", ascending=True)

    # Create visualizations using plotly
    figs = []

    # 1. Time breakdown bar chart
    fig1 = px.bar(
        df,
        y="Function",
        x=["Total Time (s)", "Cumulative (s)"],
        orientation="h",
        title="Time Breakdown by Function",
        barmode="overlay",
        opacity=0.7,
        template="plotly_dark",
    )
    fig1.update_layout(
        plot_bgcolor="#1f2630",
        paper_bgcolor="#1f2630",
        font={"color": "#ffffff"},
        height=max(400, len(df) * 30),
    )
    figs.append(fig1)

    # 2. Time per call scatter
    fig2 = px.scatter(
        df,
        x="Calls",
        y="Time/Call (ms)",
        size="Total Time (s)",
        hover_data=["Function"],
        color="Time %",
        color_continuous_scale="Viridis",
        title="Time per Call vs Number of Calls",
        template="plotly_dark",
    )
    fig2.update_layout(
        plot_bgcolor="#1f2630", paper_bgcolor="#1f2630", font={"color": "#ffffff"}
    )
    figs.append(fig2)

    # 3. Caller relationship heatmap for top functions
    top_n = min(20, len(df))
    fig3 = px.density_heatmap(
        df.head(top_n),
        x="Module",
        y="Function",
        z="Time %",
        title="Module-Function Time Distribution (Top 20)",
        template="plotly_dark",
        color_continuous_scale="Viridis",
    )
    fig3.update_layout(
        xaxis_tickangle=45,
        plot_bgcolor="#1f2630",
        paper_bgcolor="#1f2630",
        font={"color": "#ffffff"},
    )
    figs.append(fig3)

    # Show summary statistics
    print("\nProfile Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Number of unique functions: {len(df)}")
    print("\nTop 5 time-consuming functions:")
    print(
        df.nlargest(5, "Total Time (s)")[
            ["Function", "Total Time (s)", "Time %", "Calls"]
        ].to_string()
    )

    # Display all figures
    for fig in figs:
        fig.show()

    return df


def collect_splits_for_tree_pair_trajectories(
    trees: List["Node"],
) -> Tuple[Dict, Dict]:
    """
    Collect per-pair trajectory distances for unique and common splits,
    then normalize these values by the *actual pairwise distance* for each tree pair,
    so the stored values become *ratios* in [0, ∞).
    Also identifies splits that are common across *all* trees.

    Returns:
        splits_container:
          A dict mapping (i, i+1) pairs to split details,
          plus a key 'all_common_splits' for splits present in all trees.

        distance_split_container:
          A dict with keys
            "unique_splits1_distances", "unique_splits2_distances",
            "common_splits_distances", "all_common_splits_distances"
          each a list of ratio values per pair.
          'all_common_splits_distances' is a single ratio for
          the set of splits common to *all* trees, normalized by
          the distance between T[0] and T[-1].
    """

    # We'll store ratio-lists, one entry per adjacent pair
    ratio_unique_splits1: List[float] = []
    ratio_unique_splits2: List[float] = []
    ratio_common_splits: List[float] = []

    # We'll compute the intersection of splits across *all* trees
    all_splits_per_tree: List[Set[Partition]] = []
    for tree in trees:
        # collect all internal splits (excluding trivial leaves)
        splits = set(tree.to_splits())
        all_splits_per_tree.append(splits)

    # Step 3: For each adjacent pair
    for i in range(len(trees) - 1):
        treeA = trees[i]
        treeB = trees[i + 1]

        splitsA = all_splits_per_tree[i]
        splitsB = all_splits_per_tree[i + 1]

        common_splits = splitsA & splitsB
        unique_splits1 = splitsA - splitsB
        unique_splits2 = splitsB - splitsA

        # sum up the raw distances for unique_splits1
        sum_unique1 = 0.0
        for us in unique_splits1:
            nodeA = treeA.find_node_by_split(us)
            if nodeA:
                sum_unique1 += circular_distance_for_node_subset(
                    target_tree=treeB,
                    target_node=nodeA,
                    reference_order=tuple(leaf.name for leaf in treeA.get_leaves()),
                )

        # sum up the raw distances for unique_splits2
        sum_unique2 = 0.0
        for us in unique_splits2:
            nodeB = treeB.find_node_by_split(us)
            if nodeB:
                sum_unique2 += circular_distance_for_node_subset(
                    target_tree=treeA,
                    target_node=nodeB,
                    reference_order=tuple(leaf.name for leaf in treeB.get_leaves()),
                )

        # sum up the raw distances for common_splits
        sum_common = 0.0
        for cs in common_splits:
            nodeA = treeA.find_node_by_split(cs)
            if nodeA:
                sum_common += circular_distance_for_node_subset(
                    target_tree=treeB,
                    target_node=nodeA,
                    reference_order=tuple(leaf.name for leaf in treeA.get_leaves()),
                )

        # Now normalize each sum by the pairwise_dist_AB to get a ratio
        ratioU1 = sum_unique1
        ratioU2 = sum_unique2
        ratioCmn = sum_common

        ratio_unique_splits1.append(ratioU1)
        ratio_unique_splits2.append(ratioU2)
        ratio_common_splits.append(ratioCmn)

    distance_split_container = {
        "unique_splits1_distances": ratio_unique_splits1,  # now these are ratio-based
        "unique_splits2_distances": ratio_unique_splits2,
        "common_splits_distances": ratio_common_splits,
    }
    # Create a placeholder splits_container (could be extended in future)
    splits_container: Dict = {}
    return (splits_container, distance_split_container)


def benchmark_comparison(
    file_path,
    optimize_both_sides=False,
    backward=True,
    iterations=20,
    midpoint_root=True,
):
    """
    Loads trees from 'file_path', then tests:
      1) Original Order
      2) Local Optimization (Classic)
      3) Global Optimal Perm (only)
      4) Global + Local (Classic)
      5) Classification-based (Unique/Sedge) only
      6) Global + Classification-based

    Plots pairwise distance trajectories and subset ratio plots using these normalized ratios.
    Also logs the total sum of pairwise distances for each method.
    """
    original_trees = read_newick(file_path)

    if midpoint_root:
        # 1) Read trees
        midpoint_rooted_trees = []
        for tree in original_trees:
            midpoint_rooted_trees.append(tree.midpoint_root())
        original_trees = midpoint_rooted_trees

    taxa = sorted({leaf.name for tree in original_trees for leaf in tree.get_leaves()})
    # We'll do up to 20 permutations for the 'global' approach
    num_permutations = 20

    # We'll store final results
    total_distances = []
    method_names = []
    pairwise_distances_list = []
    split_distance_containers_list = []

    def process_method(trees, label):
        dist_list, _ = collect_distances_for_trajectory(trees)
        if isinstance(dist_list, float):
            dist_list = [dist_list]
        sum_dist = sum(dist_list)

        total_distances.append(sum_dist)
        method_names.append(label)
        pairwise_distances_list.append(dist_list)

        # Gather ratio-based subset distances
        distance_split_container = collect_splits_for_tree_pair_trajectories(trees)
        return distance_split_container

    ################################################
    # Method 1: Original (no changes)
    ################################################

    method1_trees = [t.deep_copy() for t in original_trees]
    dist_container_m1 = process_method(method1_trees, "Method 1: Original Order")
    split_distance_containers_list.append(dist_container_m1)

    ################################################
    # Method 2: Local Optimization (Classic)
    ################################################
    method2_trees = [t.deep_copy() for t in original_trees]
    smooth_order_of_trees_classic(
        trees=method2_trees,
        rotation_functions=[
            optimize_unique_splits,
            optimize_s_edge_splits,
            # optimize_common_splits,
        ],
        n_iterations=iterations,
        optimize_two_side=optimize_both_sides,
        backward=backward,
    )
    dist_container_m2 = process_method(method2_trees, "Method 2: Local Classic")
    split_distance_containers_list.append(dist_container_m2)

    ################################################
    # Method 3: Global Optimal Perm Only
    ################################################

    random_perms = generate_permutations(taxa, sample_size=num_permutations, seed=42)
    method3_trees = [t.deep_copy() for t in original_trees]
    minimal_perm = find_minimal_distance_permutation(method3_trees, random_perms)
    # reorder all trees to that minimal perm
    for tree in method3_trees:
        tree.reorder_taxa(minimal_perm)
    dist_container_m3 = process_method(method3_trees, "Method 3: Global Perm Only")
    split_distance_containers_list.append(dist_container_m3)

    ################################################
    # Method 4: Global Perm + Local Classic
    ################################################
    method4_trees = [t.deep_copy() for t in method3_trees]  # start from the global perm
    smooth_order_of_trees_classic(
        trees=method4_trees,
        rotation_functions=[
            optimize_unique_splits,
            optimize_s_edge_splits,
            # optimize_common_splits,
        ],
        n_iterations=iterations,
        optimize_two_side=optimize_both_sides,
        backward=backward,
    )
    dist_container_m4 = process_method(
        method4_trees, "Method 4: Global + Local Classic"
    )
    split_distance_containers_list.append(dist_container_m4)

    ################################################
    # Method 5: Classification-based Only
    ################################################
    method5_trees = [t.deep_copy() for t in original_trees]
    smooth_order_unique_sedge(
        trees=method5_trees,
        n_iterations=iterations,
        backward=backward,
    )
    dist_container_m5 = process_method(method5_trees, "Method 5: Classification Only")
    split_distance_containers_list.append(dist_container_m5)

    ################################################
    # Method 6: Global + Classification
    ################################################
    method6_trees = [
        t.deep_copy() for t in method3_trees
    ]  # again start from global perm
    smooth_order_unique_sedge(
        trees=method6_trees,
        n_iterations=iterations,
        backward=backward,
    )
    dist_container_m6 = process_method(
        method6_trees, "Method 6: Global + Classification"
    )
    split_distance_containers_list.append(dist_container_m6)

    ################################################
    # Generate distance plots
    ################################################

    plot_tree_trajectories_scatter(
        pairwise_distances_list=pairwise_distances_list,
        method_names=method_names,
    )

    # Now these ratio-based distances in each distance_split_container
    # are displayed by plot_tree_subset_ratios
    plot_tree_subset_ratios(
        method_names=method_names,
        pairwise_distances_list=pairwise_distances_list,
        split_distance_containers_list=split_distance_containers_list,
    )