"""
Analysis module for benchmark utilities.

This module contains functionality for analyzing tree splits, distances,
and other tree-related metrics used in benchmarking.
"""

from typing import List, Set, Tuple, Dict
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition
from brancharchitect.distances.distances import (
    relative_robinson_foulds_distance,
    calculate_along_trajectory,
)


def collect_splits_for_tree_pair_trajectories(
    trees: List[Node],
) -> Tuple[Dict, Dict[str, List[float]]]:
    """
    Collect per-pair trajectory distances for unique and common splits.
    
    Parameters
    ----------
    trees : List[Node]
        List of trees to analyze
        
    Returns
    -------
    Tuple[Dict, Dict[str, List[float]]]
        Tuple containing splits container and distance split container
    """
    ratio_unique_splits1: List[float] = []
    ratio_unique_splits2: List[float] = []
    ratio_common_splits: List[float] = []
    all_splits_per_tree: List[Set[Partition]] = []

    # Collect splits for each tree
    for tree in trees:
        splits = set(tree.to_splits())
        all_splits_per_tree.append(splits)

    # Compare adjacent tree pairs
    for i in range(len(trees) - 1):
        treeA = trees[i]
        treeB = trees[i + 1]
        splitsA: Set[Partition] = all_splits_per_tree[i]
        splitsB: Set[Partition] = all_splits_per_tree[i + 1]
        
        # Calculate set differences and intersections
        common_splits: Set[Partition] = splitsA & splitsB
        unique_splits1: Set[Partition] = splitsA - splitsB
        unique_splits2: Set[Partition] = splitsB - splitsA

        # Calculate distances for unique splits in tree A
        sum_unique1 = 0.0
        for us in unique_splits1:
            nodeA: Node | None = treeA.find_node_by_split(us)
            if nodeA:
                sum_unique1 += 1.0  # Placeholder: replace with your distance function

        # Calculate distances for unique splits in tree B
        sum_unique2 = 0.0
        for us in unique_splits2:
            nodeB: Node | None = treeB.find_node_by_split(us)
            if nodeB:
                sum_unique2 += 1.0  # Placeholder

        # Calculate distances for common splits
        sum_common = 0.0
        for cs in common_splits:
            nodeA = treeA.find_node_by_split(cs)
            if nodeA:
                sum_common += 1.0  # Placeholder

        # Store calculated ratios
        ratio_unique_splits1.append(sum_unique1)
        ratio_unique_splits2.append(sum_unique2)
        ratio_common_splits.append(sum_common)

    distance_split_container: Dict[str, List[float]] = {
        "unique_splits1_distances": ratio_unique_splits1,
        "unique_splits2_distances": ratio_unique_splits2,
        "common_splits_distances": ratio_common_splits,
    }
    
    # Empty splits container for compatibility
    splits_container: Dict = {}
    
    return (splits_container, distance_split_container)


def process_benchmark_method(
    trees: List[Node], 
    label: str,
    collect_distances_func
) -> Tuple[float, List[float], Dict[str, List[float]]]:
    """
    Process a single benchmark method and return results.
    
    Parameters
    ----------
    trees : List[Node]
        List of trees to process
    label : str
        Label for the method
    collect_distances_func : callable
        Function to collect distances from tree trajectory
        
    Returns
    -------
    Tuple[float, List[float], Dict[str, List[float]]]
        Tuple containing total distance, distance list, and split distance container
    """
    # Collect distances for trajectory
    dist_list, _ = collect_distances_func(trees)
    if isinstance(dist_list, float):
        dist_list = [dist_list]
    
    sum_dist = sum(dist_list)
    
    # Collect split analysis
    _, distance_split_container = collect_splits_for_tree_pair_trajectories(trees)
    
    return sum_dist, dist_list, distance_split_container


def calculate_robinson_foulds_distances(
    trees: List[Node]
) -> List[float]:
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
    
    return calculate_along_trajectory(trees, relative_robinson_foulds_distance)


def calculate_split_statistics(
    trees: List[Node]
) -> Dict[str, any]:
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
        "common_splits_ratio": len(common_splits) / len(all_unique_splits) if all_unique_splits else 0
    }