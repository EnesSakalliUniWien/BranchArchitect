"""Utility functions for tree pair iterations and interpolation."""

from typing import List, Tuple, Any
from brancharchitect.tree import Node

def iter_consecutive_pairs(trees: List[Node]) -> List[Tuple[int, Node, Node, bool, bool]]:
    """
    Iterate over consecutive pairs of trees with metadata.

    Args:
        trees: List of phylogenetic trees.

    Yields:
        Tuples of (pair_index, source_tree, target_tree, is_first, is_last).
    """
    if len(trees) < 2:
        return

    for i in range(len(trees) - 1):
        yield (
            i,
            trees[i],
            trees[i+1],
            i == 0,
            i == len(trees) - 2
        )
