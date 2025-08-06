import logging
from typing import List

from brancharchitect.io import read_newick
from brancharchitect.tree import Node
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.leaforder.tree_order_optimisation_global import (
    collect_distances_for_trajectory,
    generate_permutations,
    find_minimal_distance_permutation,
)
from brancharchitect.rooting.rooting import (
    reroot_to_compared_tree,
)

from .config import (
    BENCHMARK_COMBINATIONS,
    DEFAULT_N_ITERATIONS,
    DEFAULT_NUM_PERMUTATIONS,
)
from .profiling import run_profiler
from .analysis import (
    process_benchmark_method,
    calculate_robinson_foulds_distances,
)
from .benchmark_visualisation import (
    plot_tree_trajectories_scatter,
    plot_tree_subset_ratios,
    plot_robinson_foulds_trajectory,
)

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def profile_and_visualize(
    filepath: str,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    bidirectional: bool = True,
    min_time_percent: float = 1.0,
    focus_paths=None,
    with_rooting: bool = True,
):
    """
    Profile and visualize the TreeOrderOptimizer on a set of trees.

    Parameters
    ----------
    filepath : str
        Path to the tree file
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization
    min_time_percent : float
        Minimum time percentage to include in profiling results
    focus_paths : Optional[List[str]]
        List of path patterns to focus on in profiling
    with_rooting : bool
        Whether to apply rerooting

    Returns
    -------
    pd.DataFrame
        DataFrame containing profiling results
    """
    return run_profiler(
        benchmark_comparison,
        file_path=filepath,
        n_iterations=n_iterations,
        bidirectional=bidirectional,
        with_rooting=with_rooting,
        min_time_percent=min_time_percent,
        focus_paths=focus_paths,
    )


def benchmark_comparison(
    file_path: str,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    bidirectional: bool = True,
    with_rooting: bool = True,
):
    """
    Comprehensive benchmark comparison of different tree ordering methods.

    This function loads trees from 'file_path', then tests:
      1) Original Order
      2) Fiedler Order Only
      3) TreeOrderOptimizer (local optimization) - forward and bidirectional
      4) Fiedler + TreeOrderOptimizer combinations
      5) Global Optimal Permutation (only)
      6) Global + TreeOrderOptimizer

    Parameters
    ----------
    file_path : str
        Path to the tree file
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization for final methods
    with_rooting : bool
        Whether to apply rerooting for consistency
    """
    original_trees_result: Node | List[Node] = read_newick(file_path)

    # Ensure we have a list of trees
    if isinstance(original_trees_result, Node):
        original_trees = [original_trees_result]
    else:
        original_trees = original_trees_result

    taxa = sorted({leaf.name for tree in original_trees for leaf in tree.get_leaves()})
    num_permutations = DEFAULT_NUM_PERMUTATIONS

    total_distances = []
    method_names = []
    pairwise_distances_list = []
    split_distance_containers_list = []
    robinson_foulds_data = {}  # Store Robinson-Foulds distances for plotting

    # Apply rerooting if requested
    if with_rooting:
        for i in range(len(original_trees) - 1):
            original_trees[i] = reroot_to_compared_tree(
                original_trees[i + 1], original_trees[i]
            )

    # Calculate baseline Robinson-Foulds distances for original trees
    print("=== BASELINE ROBINSON-FOULDS ANALYSIS ===")
    rf_distances = calculate_robinson_foulds_distances(original_trees)
    total_rf = 0.0  # Initialize baseline
    if rf_distances:
        total_rf = sum(rf_distances)
        avg_rf = total_rf / len(rf_distances)
        max_rf = max(rf_distances)
        min_rf = min(rf_distances)
        print(f"Original tree trajectory Robinson-Foulds distances:")
        print(f"  Total: {total_rf:.4f}")
        print(f"  Average: {avg_rf:.4f}")
        print(f"  Max: {max_rf:.4f}")
        print(f"  Min: {min_rf:.4f}")
        print(f"  Individual distances: {[f'{d:.4f}' for d in rf_distances]}")
    else:
        print("Not enough trees for Robinson-Foulds distance calculation")
    print("=" * 45)

    # Process all benchmark combinations
    for combo in BENCHMARK_COMBINATIONS:
        trees = [t.deep_copy() for t in original_trees]
        optimizer = TreeOrderOptimizer(trees)

        if combo["fiedler"]:
            optimizer.apply_fiedler_ordering()
        if combo["optimizer"]:
            optimizer.optimize(
                n_iterations=n_iterations, bidirectional=combo["bidirectional"]
            )

        sum_dist, dist_list, dist_container = process_benchmark_method(
            trees, combo["label"], collect_distances_for_trajectory
        )

        # Calculate Robinson-Foulds distances for this method
        method_rf_distances = calculate_robinson_foulds_distances(trees)
        robinson_foulds_data[combo["label"]] = method_rf_distances  # Store for plotting
        if method_rf_distances:
            total_method_rf = sum(method_rf_distances)
            print(f"{combo['label']}: {combo['explanation']}")
            print(
                f"  Robinson-Foulds total: {total_method_rf:.4f} (vs baseline: {total_rf:.4f})"
            )
        else:
            print(f"{combo['label']}: {combo['explanation']}")

        total_distances.append(sum_dist)
        method_names.append(combo["label"])
        pairwise_distances_list.append(dist_list)
        split_distance_containers_list.append(dist_container)

    # Method: Global Optimal Permutation Only
    random_perms = generate_permutations(taxa, sample_size=num_permutations, seed=42)
    method3_trees = [t.deep_copy() for t in original_trees]
    minimal_perm = find_minimal_distance_permutation(method3_trees, random_perms)
    for tree in method3_trees:
        tree.reorder_taxa(minimal_perm)

    sum_dist_m3, dist_list_m3, dist_container_m3 = process_benchmark_method(
        method3_trees, "Global Perm Only", collect_distances_for_trajectory
    )

    # Calculate Robinson-Foulds distances for global permutation method
    method3_rf_distances = calculate_robinson_foulds_distances(method3_trees)
    robinson_foulds_data["Global Perm Only"] = (
        method3_rf_distances  # Store for plotting
    )
    if method3_rf_distances:
        total_method3_rf = sum(method3_rf_distances)
        print(
            f"Global Perm Only: Robinson-Foulds total: {total_method3_rf:.4f} (vs baseline: {total_rf:.4f})"
        )

    total_distances.append(sum_dist_m3)
    method_names.append("Global Perm Only")
    pairwise_distances_list.append(dist_list_m3)
    split_distance_containers_list.append(dist_container_m3)

    # Method: Global + TreeOrderOptimizer
    method4_trees = [t.deep_copy() for t in method3_trees]
    optimizer4 = TreeOrderOptimizer(method4_trees)
    optimizer4.optimize(n_iterations=n_iterations, bidirectional=bidirectional)

    sum_dist_m4, dist_list_m4, dist_container_m4 = process_benchmark_method(
        method4_trees, "Global + TreeOrderOptimizer", collect_distances_for_trajectory
    )

    # Calculate Robinson-Foulds distances for global + optimizer method
    method4_rf_distances = calculate_robinson_foulds_distances(method4_trees)
    robinson_foulds_data["Global + TreeOrderOptimizer"] = (
        method4_rf_distances  # Store for plotting
    )
    if method4_rf_distances:
        total_method4_rf = sum(method4_rf_distances)
        print(
            f"Global + TreeOrderOptimizer: Robinson-Foulds total: {total_method4_rf:.4f} (vs baseline: {total_rf:.4f})"
        )

    total_distances.append(sum_dist_m4)
    method_names.append("Global + TreeOrderOptimizer")
    pairwise_distances_list.append(dist_list_m4)
    split_distance_containers_list.append(dist_container_m4)

    # Generate visualizations
    # plot_tree_trajectories_scatter(
    #     pairwise_distances_list=pairwise_distances_list,
    #     method_names=method_names,
    # )

    # plot_tree_subset_ratios(
    #     method_names=method_names,
    #     pairwise_distances_list=pairwise_distances_list,
    #     split_distance_containers_list=split_distance_containers_list,
    # )

    # Generate Robinson-Foulds trajectory plot
    print("\n=== ROBINSON-FOULDS DISTANCE VISUALIZATION ===")
    plot_robinson_foulds_trajectory(
        pairwise_distances_list=pairwise_distances_list,
        method_names=method_names,
        robinson_foulds_data=robinson_foulds_data,
    )
