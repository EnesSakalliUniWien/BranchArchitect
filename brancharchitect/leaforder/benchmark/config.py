"""
Configuration module for benchmark utilities.

This module contains all benchmark configurations, method combinations,
and constants used across the benchmark system.
"""

from typing import List, Dict, Any

# Extensive Benchmark: All Fiedler/Optimizer/Bidirectional Combinations
BENCHMARK_COMBINATIONS: List[Dict[str, Any]] = [
    {
        "label": "Original Order",
        "fiedler": False,
        "optimizer": False,
        "bidirectional": False,
        "explanation": "No reordering or optimization; baseline.",
    },
    {
        "label": "Fiedler Order Only",
        "fiedler": True,
        "optimizer": False,
        "bidirectional": False,
        "explanation": "Applies consensus Fiedler (spectral) ordering to all trees, but no further optimization.",
    },
    {
        "label": "TreeOrderOptimizer Local (forward)",
        "fiedler": False,
        "optimizer": True,
        "bidirectional": False,
        "explanation": "Runs the local optimizer in forward mode only (left-to-right).",
    },
    {
        "label": "TreeOrderOptimizer Local (bidirectional)",
        "fiedler": False,
        "optimizer": True,
        "bidirectional": True,
        "explanation": "Runs the local optimizer in both directions (left-to-right and right-to-left).",
    },
    {
        "label": "Fiedler + TreeOrderOptimizer (forward)",
        "fiedler": True,
        "optimizer": True,
        "bidirectional": False,
        "explanation": "Applies Fiedler ordering, then runs the local optimizer in forward mode.",
    },
    {
        "label": "Fiedler + TreeOrderOptimizer (bidirectional)",
        "fiedler": True,
        "optimizer": True,
        "bidirectional": True,
        "explanation": "Applies Fiedler ordering, then runs the local optimizer in both directions.",
    },
]

# Default benchmark parameters
DEFAULT_N_ITERATIONS = 20
DEFAULT_NUM_PERMUTATIONS = 20
DEFAULT_MIN_TIME_PERCENT = 1.0
DEFAULT_WINDOW_SIZE = 10
DEFAULT_EPSILON = 1e-9

# Profiling configuration
PROFILING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s %(levelname)s: %(message)s",
}

# Visualization theme configurations
COLORBLIND_FRIENDLY_COLORS = {
    "blue": "#0072B2",
    "pink": "#CC79A7", 
    "dark_gray": "#777777"
}

PASTEL_COLORS = {
    "bg_color": "#F5EEE6",
    "text_color": "#5A4F4F",
    "grid_color": "#C8BCB1",
    "palette": ["#7FB5B5", "#D8A5B3", "#B39CD0", "#E1C16E"]
}

VIBRANT_COLORS = {
    "unique1": "#FFD700",
    "unique2": "#FF69B4", 
    "common": "#32CD32"
}

# Plot styling configurations
DEFAULT_PLOT_CONFIG = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "gray",
    "grid.color": "gray",
    "grid.alpha": 0.7,
    "font.size": 10,
    "axes.titlepad": 10,
}

VIBRANT_PLOT_CONFIG = {
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