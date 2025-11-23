from typing import List, Tuple, Union
import functools
from brancharchitect.tree import Node

##################################################
#                Distance Utilities
##################################################


def _circular_distance_cached(x: Tuple[int, ...], y: Tuple[int, ...]) -> float:
    """
    Helper for normalized circular distance between two integer-ranked tuples x and y.
    Uses the formula: distance = sum( min(|pos_x - pos_y|, n - |pos_x - pos_y|) )
      / ( n * (n//2) )
    """
    n = len(x)
    index_x = {elem: i for i, elem in enumerate(x)}
    index_y = {elem: i for i, elem in enumerate(y)}
    distance = 0
    for element in y:
        diff = abs(index_x[element] - index_y[element])
        distance += min(diff, n - diff)
    max_possible_distance = n * (n // 2)
    return distance / max_possible_distance


@functools.lru_cache(maxsize=None)
def create_ranks(
    reference_items: Tuple[str, ...], items: Tuple[str, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert two string-based orders into integer rank tuples for faster distance calc.
    """
    item_to_rank = {item: idx for idx, item in enumerate(reference_items)}
    original_rank = tuple(range(len(reference_items)))
    target_rank = tuple(item_to_rank[element] for element in items)
    return original_rank, target_rank


@functools.lru_cache(maxsize=None)
def circular_distance(order_x: tuple[str, ...], order_y: tuple[str, ...]) -> float:
    """
    Normalized circular distance between two tuple orders of leaves.
    """
    if not order_x or not order_y:
        raise ValueError("Input tuples must not be empty.")
    if len(set(order_x)) != len(order_x) or len(set(order_y)) != len(order_y):
        raise ValueError("Input tuples must contain unique elements.")
    if set(order_x) != set(order_y):
        raise ValueError(
            "The two orders must contain the same set of elements.",
            order_x,
            "\n",
            order_y,
            "\n",
        )
    rank_x, rank_y = create_ranks(order_x, order_y)
    return _circular_distance_cached(rank_x, rank_y)


def circular_distance_tree_pair(reference_tree: Node, target_tree: Node) -> float:
    """
    Circular distance between the current leaf orders of two trees.
    """
    ox: tuple[str, ...] = reference_tree.get_current_order()
    oy: tuple[str, ...] = target_tree.get_current_order()
    if len(set(ox)) != len(ox):
        raise ValueError(f"Reference tree order contains duplicates: {ox}")
    if len(set(oy)) != len(oy):
        raise ValueError(f"Target tree order contains duplicates: {oy}")
    return circular_distance(ox, oy)


def circular_distance_based_on_reference(
    target_tree: Node, reference_order: tuple[str, ...]
) -> float:
    """
    Full-tree distance vs. a known reference_order (tuple).
    """

    tree_order: tuple[str, ...] = target_tree.get_current_order()
    return circular_distance(tuple(reference_order), tuple(tree_order))


def circular_distance_for_node_subset(
    target_tree: Node, reference_order: tuple[str, ...], target_node: Node
) -> float:
    """
    Circular distance for only the subset of leaves under target_node in target_tree.
    """
    node_leaves: List[Node] = target_node.get_leaves()
    node_leaf_names: set[str | None] = {leaf.name for leaf in node_leaves}
    filtered_ref = tuple(n for n in reference_order if n in node_leaf_names)
    full_target: tuple[str, ...] = target_tree.get_current_order()
    filtered_target = tuple(n for n in full_target if n in node_leaf_names)
    return circular_distance(filtered_ref, filtered_target)


def circular_distances_trees(
    trees: List[Node], return_pairwise: bool = False
) -> Union[float, list[float]]:
    """
    Total circular distance for a list of trees. Optionally return pairwise distances.
    """
    total_dist: float = 0.0
    pairwise: list[float] = []
    for i in range(0, len(trees) - 1, 1):
        # Pass the taxa order, not the Node, as reference_order
        d: float = circular_distance_based_on_reference(
            trees[i + 1], trees[i].get_current_order()
        )
        total_dist += d
        pairwise.append(d)
    if return_pairwise:
        return pairwise
    return total_dist / (len(trees) - 1)
