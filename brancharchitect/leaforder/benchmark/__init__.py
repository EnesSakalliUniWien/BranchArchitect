"""
Benchmark package for tree order optimization.

This package provides modular benchmarking utilities for evaluating
tree ordering algorithms and optimization strategies.

Modules:
    config: Configuration constants and benchmark method definitions
    profiling: Performance profiling and analysis tools
    analysis: Tree split analysis and distance calculations
    benchmark_utilities: Main benchmarking functions
    benchmark_visualisation: Visualization utilities for benchmark results
"""

from .config import (
    BENCHMARK_COMBINATIONS,
    DEFAULT_N_ITERATIONS,
    DEFAULT_NUM_PERMUTATIONS,
    DEFAULT_MIN_TIME_PERCENT,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_EPSILON,
    PROFILING_CONFIG,
    COLORBLIND_FRIENDLY_COLORS,
    PASTEL_COLORS,
    VIBRANT_COLORS,
    DEFAULT_PLOT_CONFIG,
    VIBRANT_PLOT_CONFIG,
)

from .profiling import (
    configure_logging,
    create_profile_dataframe,
    create_profiling_visualizations,
    print_profile_summary,
    run_profiler,
)

from .analysis import (
    collect_splits_for_tree_pair_trajectories,
    process_benchmark_method,
    calculate_split_statistics,
    calculate_robinson_foulds_distances,
)

from .benchmark_utilities import (
    profile_and_visualize,
    benchmark_comparison,
)

from .benchmark_visualisation import (
    plot_tree_trajectories_scatter,
    plot_tree_subset_ratios,
    plot_method_and_permutation_boxplots,
    plot_robinson_foulds_trajectory,
)

__all__ = [
    # Config
    "BENCHMARK_COMBINATIONS",
    "DEFAULT_N_ITERATIONS",
    "DEFAULT_NUM_PERMUTATIONS", 
    "DEFAULT_MIN_TIME_PERCENT",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_EPSILON",
    "PROFILING_CONFIG",
    "COLORBLIND_FRIENDLY_COLORS",
    "PASTEL_COLORS",
    "VIBRANT_COLORS",
    "DEFAULT_PLOT_CONFIG",
    "VIBRANT_PLOT_CONFIG",
    
    # Profiling
    "configure_logging",
    "create_profile_dataframe",
    "create_profiling_visualizations",
    "print_profile_summary",
    "run_profiler",
    
    # Analysis
    "collect_splits_for_tree_pair_trajectories",
    "process_benchmark_method",
    "calculate_split_statistics",
    "calculate_robinson_foulds_distances",
    
    # Main utilities
    "profile_and_visualize",
    "benchmark_comparison",
    
    # Visualization
    "plot_tree_trajectories_scatter",
    "plot_tree_subset_ratios", 
    "plot_method_and_permutation_boxplots",
    "plot_robinson_foulds_trajectory",
]