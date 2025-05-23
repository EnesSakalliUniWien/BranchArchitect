import logging
import cProfile
import pstats
import pandas as pd
import plotly.express as px
from typing import List, Set, Tuple, Dict

from brancharchitect.io import read_newick
from brancharchitect.tree import Node
from brancharchitect.partition_set import Partition
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.leaforder.tree_order_optimisation_global import (
    collect_distances_for_trajectory,
    generate_permutations,
    find_minimal_distance_permutation,
)
from brancharchitect.leaforder.benchmark_visualisation import (
    plot_tree_trajectories_scatter,
    plot_tree_subset_ratios,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_and_visualize(
    filepath,
    n_iterations=20,
    bidirectional=True,
    min_time_percent=1.0,
    focus_paths=None,
):
    """
    Profile and visualize the TreeOrderOptimizer on a set of trees.
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
            n_iterations=n_iterations,
            bidirectional=bidirectional,
        )
        profiler.disable()
        logging.info("Profiling completed successfully.")
    except Exception as e:
        profiler.disable()
        logging.error(f"An error occurred during profiling: {e}")
        raise e

    stats = pstats.Stats(profiler)
    total_time = stats.total_tt
    profile_data = []

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        time_percent = (tt / total_time) * 100
        time_per_call = tt / cc if cc > 0 else 0
        if time_percent < min_time_percent:
            continue
        if focus_paths:
            if not any(path in func[0] for path in focus_paths):
                continue
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
    figs = []

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

    print("\nProfile Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Number of unique functions: {len(df)}")
    print("\nTop 5 time-consuming functions:")
    print(
        df.nlargest(5, "Total Time (s)")[
            ["Function", "Total Time (s)", "Time %", "Calls"]
        ].to_string()
    )

    for fig in figs:
        fig.show()

    return df


def collect_splits_for_tree_pair_trajectories(
    trees: List[Node],
) -> Tuple[Dict, Dict]:
    # Collect per-pair trajectory distances for unique and common splits
    ratio_unique_splits1: List[float] = []
    ratio_unique_splits2: List[float] = []
    ratio_common_splits: List[float] = []
    all_splits_per_tree: List[Set[Partition]] = []

    for tree in trees:
        splits = set(tree.to_splits())
        all_splits_per_tree.append(splits)

    for i in range(len(trees) - 1):
        treeA = trees[i]
        treeB = trees[i + 1]
        splitsA = all_splits_per_tree[i]
        splitsB = all_splits_per_tree[i + 1]
        common_splits = splitsA & splitsB
        unique_splits1 = splitsA - splitsB
        unique_splits2 = splitsB - splitsA

        sum_unique1 = 0.0
        for us in unique_splits1:
            nodeA = treeA.find_node_by_split(us)
            if nodeA:
                sum_unique1 += 1.0  # Placeholder: replace with your distance function

        sum_unique2 = 0.0
        for us in unique_splits2:
            nodeB = treeB.find_node_by_split(us)
            if nodeB:
                sum_unique2 += 1.0  # Placeholder

        sum_common = 0.0
        for cs in common_splits:
            nodeA = treeA.find_node_by_split(cs)
            if nodeA:
                sum_common += 1.0  # Placeholder

        ratioU1 = sum_unique1
        ratioU2 = sum_unique2
        ratioCmn = sum_common
        ratio_unique_splits1.append(ratioU1)
        ratio_unique_splits2.append(ratioU2)
        ratio_common_splits.append(ratioCmn)

    distance_split_container = {
        "unique_splits1_distances": ratio_unique_splits1,
        "unique_splits2_distances": ratio_unique_splits2,
        "common_splits_distances": ratio_common_splits,
    }
    splits_container: Dict = {}
    return (splits_container, distance_split_container)


def benchmark_comparison(
    file_path,
    n_iterations=20,
    bidirectional=True,
):
    """
    Loads trees from 'file_path', then tests:
      1) Original Order
      1.5) Fiedler Order Only
      2) TreeOrderOptimizer (local optimization)
      3) Global Optimal Perm (only)
      4) Global + TreeOrderOptimizer
    """
    original_trees = read_newick(file_path)

    taxa = sorted({leaf.name for tree in original_trees for leaf in tree.get_leaves()})
    num_permutations = 20

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
        # Only use the second element (distance_split_container) for plotting
        _, distance_split_container = collect_splits_for_tree_pair_trajectories(trees)
        return distance_split_container

    # --- Extensive Benchmark: All Fiedler/Optimizer/Bidirectional Combinations ---
    combinations = [
        {"label": "Original Order", "fiedler": False, "optimizer": False, "bidirectional": False, "explanation": "No reordering or optimization; baseline."},
        {"label": "Fiedler Order Only", "fiedler": True, "optimizer": False, "bidirectional": False, "explanation": "Applies consensus Fiedler (spectral) ordering to all trees, but no further optimization."},
        {"label": "TreeOrderOptimizer Local (forward)", "fiedler": False, "optimizer": True, "bidirectional": False, "explanation": "Runs the local optimizer in forward mode only (left-to-right)."},
        {"label": "TreeOrderOptimizer Local (bidirectional)", "fiedler": False, "optimizer": True, "bidirectional": True, "explanation": "Runs the local optimizer in both directions (left-to-right and right-to-left)."},
        {"label": "Fiedler + TreeOrderOptimizer (forward)", "fiedler": True, "optimizer": True, "bidirectional": False, "explanation": "Applies Fiedler ordering, then runs the local optimizer in forward mode."},
        {"label": "Fiedler + TreeOrderOptimizer (bidirectional)", "fiedler": True, "optimizer": True, "bidirectional": True, "explanation": "Applies Fiedler ordering, then runs the local optimizer in both directions."},
    ]

    for combo in combinations:
        trees = [t.deep_copy() for t in original_trees]
        optimizer = TreeOrderOptimizer(trees)
        if combo["fiedler"]:
            optimizer.apply_fiedler_ordering()
        if combo["optimizer"]:
            optimizer.optimize(n_iterations=n_iterations, bidirectional=combo["bidirectional"])
        dist_container = process_method(trees, combo["label"])
        split_distance_containers_list.append(dist_container)
        # Optionally print explanation for each method
        print(f"{combo['label']}: {combo['explanation']}")

    # Method 3: Global Optimal Perm Only
    random_perms = generate_permutations(taxa, sample_size=num_permutations, seed=42)
    method3_trees = [t.deep_copy() for t in original_trees]
    minimal_perm = find_minimal_distance_permutation(method3_trees, random_perms)
    for tree in method3_trees:
        tree.reorder_taxa(minimal_perm)
    dist_container_m3 = process_method(method3_trees, "Global Perm Only")
    split_distance_containers_list.append(dist_container_m3)

    # Method 4: Global + TreeOrderOptimizer
    method4_trees = [t.deep_copy() for t in method3_trees]
    optimizer4 = TreeOrderOptimizer(method4_trees)
    optimizer4.optimize(n_iterations=n_iterations, bidirectional=bidirectional)
    dist_container_m4 = process_method(method4_trees, "Global + TreeOrderOptimizer")
    split_distance_containers_list.append(dist_container_m4)

    plot_tree_trajectories_scatter(
        pairwise_distances_list=pairwise_distances_list,
        method_names=method_names,
    )

    plot_tree_subset_ratios(
        method_names=method_names,
        pairwise_distances_list=pairwise_distances_list,
        split_distance_containers_list=split_distance_containers_list,
    )