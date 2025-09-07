"""
Results processing and visualization utilities.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def print_baseline_analysis(rf_distances: List[float], total_rf: float) -> None:
    """
    Print baseline Robinson-Foulds analysis.

    Parameters
    ----------
    rf_distances : List[float]
        List of Robinson-Foulds distances
    total_rf : float
        Total Robinson-Foulds distance
    """
    print("=== BASELINE ROBINSON-FOULDS ANALYSIS ===")
    if rf_distances:
        avg_rf = total_rf / len(rf_distances)
        max_rf = max(rf_distances)
        min_rf = min(rf_distances)
        print("Original tree trajectory Robinson-Foulds distances:")
        print(f"  Total: {total_rf:.4f}")
        print(f"  Average: {avg_rf:.4f}")
        print(f"  Max: {max_rf:.4f}")
        print(f"  Min: {min_rf:.4f}")
        print(f"  Individual distances: {[f'{d:.4f}' for d in rf_distances]}")
    else:
        print("Not enough trees for Robinson-Foulds distance calculation")
    print("=" * 45)


def print_method_comparison(
    method_name: str,
    explanation: str,
    method_rf_distances: List[float],
    total_rf: float,
) -> None:
    """
    Print comparison for a specific method.

    Parameters
    ----------
    method_name : str
        Name of the method
    explanation : str
        Method explanation
    method_rf_distances : List[float]
        Method's Robinson-Foulds distances
    total_rf : float
        Baseline total Robinson-Foulds distance
    """
    if method_rf_distances:
        total_method_rf = sum(method_rf_distances)
        print(f"{method_name}: {explanation}")
        print(
            f"  Robinson-Foulds total: {total_method_rf:.4f} (vs baseline: {total_rf:.4f})"
        )
    else:
        print(f"{method_name}: {explanation}")


def aggregate_results(
    combo_results: tuple, global_perm_results: tuple, global_plus_opt_results: tuple
) -> tuple:
    """
    Aggregate all benchmark results.

    Parameters
    ----------
    combo_results : tuple
        Results from benchmark combinations
    global_perm_results : tuple
        Results from global permutation benchmark
    global_plus_opt_results : tuple
        Results from global + optimizer benchmark

    Returns
    -------
    tuple
        Aggregated results
    """
    (
        total_distances,
        method_names,
        pairwise_distances_list,
        split_distance_containers_list,
        robinson_foulds_data,
    ) = combo_results

    # Add global permutation results
    (
        sum_dist_m3,
        dist_list_m3,
        dist_container_m3,
        method3_rf_distances,
    ) = global_perm_results

    total_distances.append(sum_dist_m3)
    method_names.append("Global Perm Only")
    pairwise_distances_list.append(dist_list_m3)
    split_distance_containers_list.append(dist_container_m3)
    robinson_foulds_data["Global Perm Only"] = method3_rf_distances

    # Add global + optimizer results
    (
        sum_dist_m4,
        dist_list_m4,
        dist_container_m4,
        method4_rf_distances,
    ) = global_plus_opt_results

    total_distances.append(sum_dist_m4)
    method_names.append("Global + TreeOrderOptimizer")
    pairwise_distances_list.append(dist_list_m4)
    split_distance_containers_list.append(dist_container_m4)
    robinson_foulds_data["Global + TreeOrderOptimizer"] = method4_rf_distances

    return (
        total_distances,
        method_names,
        pairwise_distances_list,
        split_distance_containers_list,
        robinson_foulds_data,
    )
