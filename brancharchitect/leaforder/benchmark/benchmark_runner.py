"""
Benchmark execution utilities.
"""

import logging
from typing import List, Dict, Any, Tuple

from brancharchitect.tree import Node
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.leaforder.tree_order_optimisation_global import (
    collect_distances_for_trajectory,
    generate_permutations,
    find_minimal_distance_permutation,
)

from .config import (
    BENCHMARK_COMBINATIONS,
    DEFAULT_NUM_PERMUTATIONS,
)
from .analysis import (
    process_benchmark_method,
    calculate_robinson_foulds_distances,
)

logger = logging.getLogger(__name__)


def run_benchmark_combinations(
    original_trees: List[Node], n_iterations: int, bidirectional: bool
) -> Tuple[
    List[float], List[str], List[List[float]], List[Any], Dict[str, List[float]]
]:
    """
    Run all benchmark combinations and collect results.

    Parameters
    ----------
    original_trees : List[Node]
        Original list of trees
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization

    Returns
    -------
    Tuple containing:
    - total_distances: List[float]
    - method_names: List[str]
    - pairwise_distances_list: List[List[float]]
    - split_distance_containers_list: List[Any]
    - robinson_foulds_data: Dict[str, List[float]]
    """
    total_distances: List[float] = []
    method_names: List[str] = []
    pairwise_distances_list: List[List[float]] = []
    split_distance_containers_list: List[Any] = []
    robinson_foulds_data: Dict[str, List[float]] = {}

    # Process all benchmark combinations
    for combo in BENCHMARK_COMBINATIONS:
        trees = [t.deep_copy() for t in original_trees]
        optimizer = TreeOrderOptimizer(trees)

        # Fiedler ordering deprecated: ignore 'fiedler' flag
        if combo["optimizer"]:
            optimizer.optimize(
                n_iterations=n_iterations, bidirectional=combo["bidirectional"]
            )

        sum_dist, dist_list, dist_container = process_benchmark_method(
            trees, combo["label"], collect_distances_for_trajectory
        )

        # Calculate Robinson-Foulds distances for this method
        method_rf_distances = calculate_robinson_foulds_distances(trees)
        robinson_foulds_data[combo["label"]] = method_rf_distances

        total_distances.append(sum_dist)
        method_names.append(combo["label"])
        pairwise_distances_list.append(dist_list)
        split_distance_containers_list.append(dist_container)

    return (
        total_distances,
        method_names,
        pairwise_distances_list,
        split_distance_containers_list,
        robinson_foulds_data,
    )


def run_global_permutation_benchmark(
    original_trees: List[Node], taxa: List[str], n_iterations: int, bidirectional: bool
) -> Tuple[float, List[float], Any, List[float]]:
    """
    Run global permutation benchmark.

    Parameters
    ----------
    original_trees : List[Node]
        Original list of trees
    taxa : List[str]
        List of taxa
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization

    Returns
    -------
    Tuple containing:
    - sum_dist: float
    - dist_list: List[float]
    - dist_container: Any
    - rf_distances: List[float]
    """
    num_permutations = DEFAULT_NUM_PERMUTATIONS
    random_perms = generate_permutations(taxa, sample_size=num_permutations, seed=42)
    method_trees = [t.deep_copy() for t in original_trees]
    minimal_perm = find_minimal_distance_permutation(method_trees, random_perms)

    if minimal_perm is not None:
        for tree in method_trees:
            tree.reorder_taxa(minimal_perm)
    else:
        logger.warning("No minimal permutation found, using original tree order")

    sum_dist, dist_list, dist_container = process_benchmark_method(
        method_trees, "Global Perm Only", collect_distances_for_trajectory
    )

    rf_distances = calculate_robinson_foulds_distances(method_trees)

    return sum_dist, dist_list, dist_container, rf_distances


def run_global_plus_optimizer_benchmark(
    base_trees: List[Node], n_iterations: int, bidirectional: bool
) -> Tuple[float, List[float], Any, List[float]]:
    """
    Run global permutation + optimizer benchmark.

    Parameters
    ----------
    base_trees : List[Node]
        Base trees (already permuted)
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization

    Returns
    -------
    Tuple containing:
    - sum_dist: float
    - dist_list: List[float]
    - dist_container: Any
    - rf_distances: List[float]
    """
    method_trees = [t.deep_copy() for t in base_trees]
    optimizer = TreeOrderOptimizer(method_trees)
    optimizer.optimize(n_iterations=n_iterations, bidirectional=bidirectional)

    sum_dist, dist_list, dist_container = process_benchmark_method(
        method_trees, "Global + TreeOrderOptimizer", collect_distances_for_trajectory
    )

    rf_distances = calculate_robinson_foulds_distances(method_trees)

    return sum_dist, dist_list, dist_container, rf_distances
