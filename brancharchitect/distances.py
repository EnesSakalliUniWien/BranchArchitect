from brancharchitect.tree import Node
from typing import List, Tuple, Dict

def sort_splits(splits: List[Tuple[int]]) -> List[Tuple[int]]:
    """
    Sorts a list of tuples. Each tuple and the list itself are sorted.
    
    Args:
    - splits (List[Tuple[int]]): A list of tuples representing splits.

    Returns:
    - List[Tuple[int]]: The sorted list of sorted tuples.
    """
    # Sort each tuple
    sorted_tuples = [tuple(sorted(split)) for split in splits]
    # Sort the list of tuples
    sorted_tuples.sort()
    return sorted_tuples

def collect_splits(tree: Node) -> Tuple[List[float], Dict[str, float]]:
    split_list : List[int] = []
    split_lengths: Dict[str, float] = {}
    _collect_splits(tree, split_list, split_lengths)
    return split_list, split_lengths

def _collect_splits(node: Node, split_list: List, split_length: dict[tuple, int]):
    split_list += [node.split_indices]
    split_length[node.split_indices] = node.length
    for child in node.children:
        _collect_splits(child, split_list, split_length)

def calculate_relative_tree_distance(tree_one: Node, tree_two: Node):
    splits_tree_one, _ = collect_splits(tree_one)    
    splits_tree_two, _ = collect_splits(tree_two)
    relative_distance = calculate_relative_split_difference(splits_tree_one, splits_tree_two)
    return relative_distance

def calculate_split_difference(list_one: List, list_two: List) -> int:
    set_one = set(map(tuple, list_one))
    set_two = set(map(tuple, list_two))

    unique_to_one = set_one - set_two
    unique_to_two = set_two - set_one
    
    difference_count = len(unique_to_one) + len(unique_to_two)
        
    return difference_count

def calculate_relative_split_difference(list_one: List, list_two: List) -> float:
    set_one = set(map(tuple, list_one))
    set_two = set(map(tuple, list_two))

    unique_to_one = set_one - set_two
    unique_to_two = set_two - set_one
    
    total_unique_differences = len(unique_to_one) + len(unique_to_two)
    total_unique_splits = len(set_one | set_two)  # Union of both sets

    # To avoid division by zero
    if total_unique_splits == 0:
        return 0.0

    relative_difference = total_unique_differences / 2
    
    return relative_difference

def calculate_relative_distances(trajectory : List[Node])-> List[float]:    
    relative_distances = []    
    for i in range(0, len(trajectory) - 1, 1):
        tree_one = trajectory[i]
        tree_two = trajectory[i+1]
        relative_distance = calculate_relative_tree_distance(tree_one, tree_two)
        relative_distances.append(relative_distance)
    return relative_distances

def calculate_weighted_distance(tree_one: Node, tree_two: Node) -> float:
    """
    Calculate the weighted Robinson-Foulds distance between two trees.

    Args:
        tree_one (Any): The first tree, in a format accepted by collect_splits.
        tree_two (Any): The second tree, in a format accepted by collect_splits.

    Returns:
        float: The weighted Robinson-Foulds distance between the two trees.
    """
    _, weights_tree_one = collect_splits(tree_one)
    _, weights_tree_two = collect_splits(tree_two)

    all_splits = set(weights_tree_one.keys()).union(set(weights_tree_two.keys()))

    weighted_distance = sum(
        abs(weights_tree_one.get(split, 0) - weights_tree_two.get(split, 0)) for split in all_splits
    )

    return weighted_distance

def calculate_weighted_distances(trajectory : List[Node])-> List[float]:
    weighted_distances = []    
    for i in range(0, len(trajectory) - 1, 1):
        tree_one = trajectory[i]
        tree_two = trajectory[i+1]
        relative_distance = calculate_weighted_distance(tree_one, tree_two)
        weighted_distances.append(relative_distance)
    return weighted_distances
