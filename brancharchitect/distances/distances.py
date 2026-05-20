from itertools import pairwise
from typing import Dict, List, Callable
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition


def relative_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    """Return the rooted-subtree symmetric difference normalized by union size.

    This is the historical backend field named ``robinson_foulds``. It is not
    the standard unrooted Robinson-Foulds bipartition distance.
    """
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
    Calculate the rooted weighted split distance between two trees.

    This uses ``Node.to_weighted_splits()``, so terminal and root-associated
    splits are included in addition to internal rooted subtrees. The frontend
    payload declares these semantics explicitly to avoid reading this as a
    standard unrooted weighted RF metric.

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
