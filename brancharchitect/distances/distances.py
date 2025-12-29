from itertools import pairwise
from typing import Dict, List, Callable, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def compute_tree_pair_component_paths(
    i: int,
    j: int,
    tree_i: Node,
    tree_j: Node,
    reroot_to_compair: bool = False,
) -> Optional[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]]:
    if i == j:
        return None  # Skip diagonal

    # Get s-edge solutions from lattice algorithm
    from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
        LatticeSolver,
    )

    raw_pivot_edge_solutions, _ = LatticeSolver(tree_i, tree_j).solve_iteratively()

    # Deduplicate solutions per pivot edge to avoid double-counting identical components
    pivot_edge_solutions = {}
    for pivot_edge, solutions in raw_pivot_edge_solutions.items():
        seen: set[int] = set()
        unique_solutions: list[Partition] = []
        for sol in solutions:
            if sol.bitmask in seen:
                continue
            seen.add(sol.bitmask)
            unique_solutions.append(sol)
        pivot_edge_solutions[pivot_edge] = unique_solutions

    # Collect components and paths, preserving multiplicity across pivot edges
    components: List[Partition] = []
    pivot_edges_for_components: List[Partition] = []
    paths_i: List[List[Node]] = []
    paths_j: List[List[Node]] = []

    for pivot_edge, solutions in pivot_edge_solutions.items():
        for component in solutions:
            # Jump path component to pivot edge
            path_i: List[Node] = jump_path_component_to_pivot_edge(
                tree=tree_i,
                component=component,
                pivot_edge_split=pivot_edge,
            )
            path_j: List[Node] = jump_path_component_to_pivot_edge(
                tree=tree_j,
                component=component,
                pivot_edge_split=pivot_edge,
            )

            components.append(component)
            pivot_edges_for_components.append(pivot_edge)
            paths_i.append(path_i)
            paths_j.append(path_j)
    return (i, j, components, pivot_edges_for_components, paths_i, paths_j)


def compute_pairwise_pivot_edge_paths(
    trees: List[Node],
) -> List[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]]:
    """Compute pivot-edge solutions and component paths for every unique tree pair.

    Returns a list of (i, j, components, pivot_edges, paths_i, paths_j) tuples for
    each successful pair (i > j), leveraging `compute_tree_pair_component_paths`
    in parallel.
    """
    for tree in tqdm(trees, desc="Preparing trees for parallel processing"):
        tree.to_splits()
        tree.build_split_index()

    pair_args = [
        (i, j, trees[i], trees[j]) for i in range(len(trees)) for j in range(i)
    ]
    # Profile the parallel computation
    results: List[Tuple[int, int, Any, Any, List[List[Node]], List[List[Node]]]] = []
    with ProcessPoolExecutor() as executor:
        future_to_pair = {
            executor.submit(compute_tree_pair_component_paths, *arg): (arg[0], arg[1])
            for arg in pair_args
        }
        for f in tqdm(as_completed(future_to_pair), total=len(future_to_pair)):
            pair_indices = future_to_pair[f]
            res = f.result()
            if res is None:
                raise RuntimeError(
                    f"compute_pair unexpectedly returned None for pair {pair_indices}"
                )
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
