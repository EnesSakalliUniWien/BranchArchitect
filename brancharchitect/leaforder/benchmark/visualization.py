"""
Visualization utilities for benchmark results.
"""

import logging
from typing import List, Dict

from .benchmark_visualisation import plot_robinson_foulds_trajectory

logger = logging.getLogger(__name__)


def create_visualizations(
    pairwise_distances_list: List[List[float]],
    method_names: List[str],
    robinson_foulds_data: Dict[str, List[float]],
) -> None:
    """
    Create and display benchmark visualizations.

    Parameters
    ----------
    pairwise_distances_list : List[List[float]]
        List of distance lists for each method
    method_names : List[str]
        Names of the methods
    robinson_foulds_data : Dict[str, List[float]]
        Robinson-Foulds data for each method
    """
    print("\n=== ROBINSON-FOULDS DISTANCE VISUALIZATION ===")
    plot_robinson_foulds_trajectory(
        pairwise_distances_list=pairwise_distances_list,
        method_names=method_names,
        robinson_foulds_data=robinson_foulds_data,
    )
