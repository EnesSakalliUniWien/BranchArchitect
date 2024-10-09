from brancharchitect.tree import Node
from typing import List, Callable
from itertools import pairwise


def robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1, splits2 = tree1.to_splits(), tree2.to_splits()
    set1 = set(splits1)
    set2 = set(splits2)
    return len(set1 ^ set2) / 2


def relative_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1, splits2 = tree1.to_splits(), tree2.to_splits()
    set1 = set(splits1)
    set2 = set(splits2)

    total_unique_differences = len(set1 ^ set2)  # Symmetric difference
    total_unique_splits = len(set1 | set2)  # Union of both sets

    # account for trivial splits
    total_unique_splits -= len(tree1._order)
    # account for split that contains all taxa
    total_unique_splits -= 1

    relative_difference = total_unique_differences / total_unique_splits

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
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()

    all_splits = set(splits1) | set(splits2)

    weighted_distance = sum(
        abs(splits1.get(split, 0) - splits2.get(split, 0)) for split in all_splits
    )

    return weighted_distance


def calculate_along_trajectory(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[float]:
    dists = [distance_function(tree1, tree2) for tree1, tree2 in pairwise(trajectory)]
    return dists


def calculate_matrix_distance(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[List[float]]:
    n = len(trajectory)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (
                i != j
            ):  # Optionally check to avoid computing distance from a node to itself
                distance_matrix[i][j] = distance_function(trajectory[i], trajectory[j])
    return distance_matrix
