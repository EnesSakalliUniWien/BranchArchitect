import math
from typing import List, Tuple
from brancharchitect.consensus_tree import get_taxa_circular_order
from brancharchitect.tree import Node

"""
####    Circular Rank Distances     ####
"""


def _circular_distance(x: List[int], y: List[int]) -> float:
    """Calculate the normalized circular distance between two ranked lists."""
    n = len(x)
    index_x = {element: idx for idx, element in enumerate(x)}
    index_y = {element: idx for idx, element in enumerate(y)}
    distance = 0
    for element in y:
        diff = abs(index_x[element] - index_y[element])
        distance += min(diff, n - diff)
    max_possible_distance = n * math.floor(n // 2)
    return distance / max_possible_distance


def create_ranks(
    reference_items: List[str], items: List[str]
) -> Tuple[List[int], List[int]]:
    """
    Create rank mappings for the given items based on the reference items.
    """
    item_to_rank = {item: idx for idx, item in enumerate(reference_items)}
    original_rank = list(range(len(reference_items)))
    reference_rank = [item_to_rank[element] for element in items]
    return original_rank, reference_rank


def circular_distance(order_x: List[str], order_y: List[str]) -> float:
    """
    Calculate the normalized circular distance between two orders.
    """
    if not order_x or not order_y:
        raise ValueError("Input lists must not be empty.")
    if len(set(order_x)) != len(order_x) or len(set(order_y)) != len(order_y):
        raise ValueError("Input lists must contain unique elements.")
    if set(order_x) != set(order_y):
        raise ValueError("The lists must contain the same elements.")
    rank_x, rank_y = create_ranks(order_x, order_y)
    return _circular_distance(rank_x, rank_y)


def circular_distance_tree_pair(t1: Node, t2: Node) -> float:
    """
    Compute the circular distance between two trees based on their taxa orders.
    """
    order_1 = get_taxa_circular_order(t1)
    order_2 = get_taxa_circular_order(t2)
    return circular_distance(order_1, order_2)


def circular_distances_trees(trees: List[Node]) -> float:
    """
    Compute the average circular distance between consecutive trees in a list.
    """
    if len(trees) < 2:
        raise ValueError("At least two trees are required to compute distances.")
    cds = [circular_distance_tree_pair(t1, t2) for t1, t2 in zip(trees[:-1], trees[1:])]
    return sum(cds) / len(cds)


