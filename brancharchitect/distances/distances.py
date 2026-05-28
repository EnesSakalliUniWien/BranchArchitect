from itertools import pairwise
from typing import Dict, List, Callable, Optional, Set
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition


def _canonical_unrooted_split(
    split: Partition, all_indices: Set[int], encoding: Dict[str, int]
) -> Optional[Partition]:
    side = set(split.indices)
    complement = all_indices - side

    if len(side) <= 1 or len(complement) <= 1:
        return None

    ordered_side = tuple(sorted(side))
    ordered_complement = tuple(sorted(complement))
    canonical_side = min(ordered_side, ordered_complement)
    return Partition(canonical_side, encoding=encoding)


def _unrooted_internal_bipartitions(tree: Node) -> Set[Partition]:
    """Return textbook RF bipartitions: internal, non-trivial, and root-invariant."""
    all_indices = set(tree.split_indices)
    return {
        canonical_split
        for split in tree.to_splits()
        if (
            canonical_split := _canonical_unrooted_split(
                split, all_indices, tree.taxa_encoding
            )
        )
        is not None
    }


def relative_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    """Return the normalized textbook Robinson-Foulds bipartition distance.

    Internal non-trivial splits are canonicalized across the root so the same
    unrooted bipartition is counted once. The raw RF count is normalized by the
    total number of internal bipartitions in both trees.
    """
    splits1: Set[Partition] = _unrooted_internal_bipartitions(tree1)
    splits2: Set[Partition] = _unrooted_internal_bipartitions(tree2)

    total_unique_differences: int = len(splits1 ^ splits2)  # Symmetric difference
    total_unique_splits: int = len(splits1) + len(splits2)

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
