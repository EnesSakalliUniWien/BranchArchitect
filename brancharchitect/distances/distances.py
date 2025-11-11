from itertools import pairwise
from typing import Dict, List, Callable, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.distances.component_distance import (
    jump_path_component_to_pivot_edge,
)
from numpy.typing import NDArray


def robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1: PartitionSet[Partition] = tree1.to_splits()
    splits2: PartitionSet[Partition] = tree2.to_splits()
    return len(splits1 ^ splits2) / 2


def relative_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1: PartitionSet[Partition] = tree1.to_splits()
    splits2: PartitionSet[Partition] = tree2.to_splits()

    total_unique_differences: int = len(splits1 ^ splits2)  # Symmetric difference
    total_unique_splits: int = len(splits1 | splits2)  # Union of both sets

    if total_unique_splits == 0:
        return 0.0
    relative_difference: float = total_unique_differences / total_unique_splits

    return relative_difference


def weighted_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    """
    Calculate the weighted Robinson-Foulds distance between two trees.

    Args:
        tree1 (Node): The first tree
        tree2 (Node): The second tree

    Returns:
        float: The weighted Robinson-Foulds distance between the two trees.
    """
    splits1: Dict[Partition, float] = tree1.to_weighted_splits()
    splits2: Dict[Partition, float] = tree2.to_weighted_splits()

    all_splits = set(splits1) | set(splits2)

    weighted_distance: float = sum(
        abs(splits1.get(split, 0) - splits2.get(split, 0)) for split in all_splits
    )

    return weighted_distance


def calculate_along_trajectory(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[float]:
    dists: List[float] = [
        distance_function(tree1, tree2) for tree1, tree2 in pairwise(trajectory)
    ]
    return dists


def calculate_matrix_distance(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[List[float]]:
    n = len(trajectory)
    distance_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (
                i != j
            ):  # Optionally check to avoid computing distance from a node to itself
                distance_matrix[i][j] = distance_function(trajectory[i], trajectory[j])
    return distance_matrix


def compute_pair(
    i: int,
    j: int,
    tree_i: Node,
    tree_j: Node,
    leaf_order: List[str] = [],
    reroot_to_compair: bool = False,
) -> Optional[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]]:
    if i == j:
        return None  # Skip diagonal

    copy_tree_i: Node = tree_i.deep_copy()
    copy_tree_j: Node = tree_j.deep_copy()

    # Get s-edge solutions from lattice algorithm
    from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
        iterate_lattice_algorithm,
    )

    pivot_edge_solutions, _ = iterate_lattice_algorithm(
        copy_tree_i, copy_tree_j, leaf_order
    )
    pivot_edges = list(pivot_edge_solutions.keys())
    # Flatten to get components (individual partitions)
    components = [
        partition
        for partitions in pivot_edge_solutions.values()
        for partition in partitions
    ]

    paths_i: List[List[Node]] = []
    paths_j: List[List[Node]] = []

    try:
        # Process each pivot edge and its solution sets
        for pivot_edge in pivot_edges:
            for component in pivot_edge_solutions[pivot_edge]:
                # Jump path component to pivot edge
                path_i: List[Node] = jump_path_component_to_pivot_edge(
                    tree=copy_tree_i,
                    component=component,
                    pivot_edge_split=pivot_edge,
                )
                path_j: List[Node] = jump_path_component_to_pivot_edge(
                    tree=copy_tree_j,
                    component=component,
                    pivot_edge_split=pivot_edge,
                )
                paths_i.append(path_i)
                paths_j.append(path_j)
    except Exception as e:
        print(f"pivot_edge_solutions structure: {pivot_edge_solutions}")
        print(f"Error while computing paths for trees {i} and {j}: {e}")
        return None
    return (i, j, components, pivot_edges, paths_i, paths_j)


def compute_all_pairs(
    trees: List[Node],
) -> List[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]]:
    pair_args = [
        (i, j, trees[i], trees[j]) for i in range(len(trees)) for j in range(i)
    ]
    # Profile the parallel computation
    results: List[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]] = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_pair, *arg) for arg in pair_args]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res is not None:
                results.append(res)
    return results


def calculate_normalised_matrix(
    results: List[Tuple[int, int, List[List[Any]], Any, List[Any], List[Any]]],
    num_trees: int,
) -> NDArray[np.float64]:
    max_component_sum: int = max(
        (sum(len(c) for c in comp) for _, _, comp, _, _, _ in results), default=1
    )
    max_num_solutions: int = max(
        (len(comp) for _, _, comp, _, _, _ in results), default=1
    )
    max_path_length: int = max(
        ((len(pi) + len(pj)) for _, _, _, _, pi, pj in results), default=1
    )
    # 2. Optionally set weights for each component (can be tuned)
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    # 3. Build normalized distance matrix
    normalized_matrix: NDArray[np.float64] = np.zeros(
        (num_trees, num_trees), dtype=float
    )
    for i, j, components, _, path_i, path_j in results:
        component_sum: int = sum(len(c) for c in components)
        num_solutions: int = len(components)
        path_lengths: int = sum([len(pi) + len(pj) for pi, pj in zip(path_i, path_j)])
        norm_component_sum: float = (
            component_sum / max_component_sum if max_component_sum else 0
        )
        norm_num_solutions: float = (
            num_solutions / max_num_solutions if max_num_solutions else 0
        )
        norm_path_length: float = (
            path_lengths / max_path_length if max_path_length else 0
        )
        dist: float = (
            w1 * norm_component_sum + w2 * norm_num_solutions + w3 * norm_path_length
        )
        normalized_matrix[i, j] = dist
        normalized_matrix[j, i] = dist

    return normalized_matrix
