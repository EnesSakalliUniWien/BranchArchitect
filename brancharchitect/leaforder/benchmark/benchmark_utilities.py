"""
Main benchmark utilities module.

This module provides the main entry points for benchmarking tree ordering methods.
It has been modularized into separate components for better maintainability.
"""

import logging
from typing import List, Optional

from .config import DEFAULT_N_ITERATIONS
from .profiling import run_profiler
from .data_loader import load_and_preprocess_trees, extract_taxa
from .benchmark_runner import (
    run_benchmark_combinations,
    run_global_permutation_benchmark,
    run_global_plus_optimizer_benchmark,
)
from .results_processor import (
    print_baseline_analysis,
    print_method_comparison,
    aggregate_results,
)
from .visualization import create_visualizations
from .analysis import calculate_robinson_foulds_distances

logger = logging.getLogger(__name__)


def profile_and_visualize(
    filepath: str,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    bidirectional: bool = True,
    min_time_percent: float = 1.0,
    focus_paths: Optional[List[str]] = None,
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
        min_time_percent=min_time_percent,
        focus_paths=focus_paths,
    )


def benchmark_comparison(
    file_path: str,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    bidirectional: bool = True,
):
    """
    Comprehensive benchmark comparison of different tree ordering methods.

    This function loads trees from 'file_path', then tests various methods
    and generates visualizations.

    Parameters
    ----------
    file_path : str
        Path to the tree file
    n_iterations : int
        Number of optimization iterations
    bidirectional : bool
        Whether to use bidirectional optimization
    """
    # Load and preprocess trees
    original_trees = load_and_preprocess_trees(file_path)
    taxa = extract_taxa(original_trees)

    # Calculate baseline Robinson-Foulds distances
    rf_distances = calculate_robinson_foulds_distances(original_trees)
    total_rf = sum(rf_distances) if rf_distances else 0.0
    print_baseline_analysis(rf_distances, total_rf)

    # Run benchmark combinations
    combo_results = run_benchmark_combinations(
        original_trees, n_iterations, bidirectional
    )

    # Run global permutation benchmark
    global_perm_results = run_global_permutation_benchmark(
        original_trees, taxa, n_iterations, bidirectional
    )

    # Run global + optimizer benchmark (using original trees as base)
    global_plus_opt_results = run_global_plus_optimizer_benchmark(
        original_trees, n_iterations, bidirectional
    )

    # Aggregate all results
    results = aggregate_results(
        combo_results, global_perm_results, global_plus_opt_results
    )
    (
        total_distances,
        method_names,
        pairwise_distances_list,
        split_distance_containers_list,
        robinson_foulds_data,
    ) = results

    # Print method comparisons
    for method_name in method_names:
        method_rf = robinson_foulds_data.get(method_name, [])
        explanation = "Benchmark method"
        print_method_comparison(method_name, explanation, method_rf, total_rf)

    # Create visualizations
    create_visualizations(pairwise_distances_list, method_names, robinson_foulds_data)
