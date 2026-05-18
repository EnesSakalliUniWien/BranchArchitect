"""
Analysis module for benchmark utilities.

This module contains functionality for analyzing tree splits, distances,
and other tree-related metrics used in benchmarking.
"""

from typing import List, Tuple, Dict, Any, Callable, Sequence, cast
from brancharchitect.tree import Node
from brancharchitect.distances.distances import (
    relative_robinson_foulds_distance,
    calculate_along_trajectory,
)


def process_benchmark_method(
    trees: List[Node],
    collect_distances_func: Callable[[List[Node]], Tuple[Sequence[float] | float, Any]],
) -> Tuple[float, List[float]]:
    """
    Process a single benchmark method and return results.

    Parameters
    ----------
    trees : List[Node]
        List of trees to process
    collect_distances_func : callable
        Function to collect distances from tree trajectory

    Returns
    -------
    Tuple[float, Sequence[float]]
        Tuple containing total distance and distance list
    """
    # Collect distances for trajectory
    dist_list, _ = collect_distances_func(trees)
    if isinstance(dist_list, float):
        dist_list = [dist_list]
    else:
        dist_list = list(dist_list)

    sum_dist = sum(dist_list)

    return sum_dist, dist_list


def calculate_robinson_foulds_distances(trees: List[Node]) -> List[float]:
    """
    Calculate relative Robinson-Foulds distances along tree trajectory.

    Parameters
    ----------
    trees : List[Node]
        List of trees to analyze

    Returns
    -------
    List[float]
        List of relative Robinson-Foulds distances between adjacent trees
    """
    if len(trees) < 2:
        return []

    return cast(
        List[float],
        calculate_along_trajectory(trees, relative_robinson_foulds_distance),
    )


def calculate_split_statistics(trees: List[Node]) -> Dict[str, Any]:
    """
    Calculate various statistics about splits in the tree collection.

    Parameters
    ----------
    trees : List[Node]
        List of trees to analyze

    Returns
    -------
    Dict[str, any]
        Dictionary containing split statistics
    """
    all_splits_per_tree = []
    for tree in trees:
        splits = set(tree.to_splits())
        all_splits_per_tree.append(splits)

    if not all_splits_per_tree:
        return {}

    # Calculate statistics
    total_splits = sum(len(splits) for splits in all_splits_per_tree)
    avg_splits_per_tree = total_splits / len(all_splits_per_tree)

    # Find unique splits across all trees
    all_unique_splits = set()
    for splits in all_splits_per_tree:
        all_unique_splits.update(splits)

    # Calculate common splits (present in all trees)
    common_splits = all_splits_per_tree[0].copy()
    for splits in all_splits_per_tree[1:]:
        common_splits &= splits

    return {
        "total_trees": len(trees),
        "total_splits": total_splits,
        "avg_splits_per_tree": avg_splits_per_tree,
        "unique_splits_count": len(all_unique_splits),
        "common_splits_count": len(common_splits),
        "common_splits_ratio": (
            len(common_splits) / len(all_unique_splits) if all_unique_splits else 0
        ),
    }
