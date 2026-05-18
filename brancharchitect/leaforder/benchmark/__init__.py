"""
Benchmark package for tree order optimization.

This package provides modular benchmarking utilities for evaluating
tree ordering algorithms and optimization strategies.

Modules:
    config: Configuration constants and benchmark method definitions
    analysis: Tree split analysis and distance calculations
    benchmark_utilities: Main benchmarking functions
    benchmark_visualisation: Optional visualization utilities for benchmark results
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

from .analysis import (
    process_benchmark_method,
    calculate_split_statistics,
    calculate_robinson_foulds_distances,
)

from .benchmark_utilities import (
    benchmark_comparison,
)

from .data_loader import (
    load_and_preprocess_trees,
    extract_taxa,
)

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
    # Analysis
    "process_benchmark_method",
    "calculate_split_statistics",
    "calculate_robinson_foulds_distances",
    # Main utilities
    "benchmark_comparison",
    # Data loading
    "load_and_preprocess_trees",
    "extract_taxa",
    # Benchmark running
    "run_benchmark_combinations",
    "run_global_permutation_benchmark",
    "run_global_plus_optimizer_benchmark",
    # Results processing
    "print_baseline_analysis",
    "print_method_comparison",
    "aggregate_results",
]
