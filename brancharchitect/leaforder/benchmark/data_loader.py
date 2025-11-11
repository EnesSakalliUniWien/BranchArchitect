"""
Data loading and preprocessing utilities for benchmarking.
"""

import logging
from typing import List

from brancharchitect.io import read_newick
from brancharchitect.tree import Node

logger = logging.getLogger(__name__)


def load_and_preprocess_trees(file_path: str) -> List[Node]:
    """
    Load trees from file and ensure they are in list format.

    Parameters
    ----------
    file_path : str
        Path to the tree file

    Returns
    -------
    List[Node]
        List of tree objects
    """
    original_trees_result = read_newick(file_path)

    # Ensure we have a list of trees
    if isinstance(original_trees_result, Node):
        original_trees = [original_trees_result]
    else:
        original_trees = original_trees_result

    return original_trees


def extract_taxa(original_trees: List[Node]) -> List[str]:
    """
    Extract sorted list of taxa from trees.

    Parameters
    ----------
    original_trees : List[Node]
        List of tree objects

    Returns
    -------
    List[str]
        Sorted list of taxa names
    """
    taxa = sorted({leaf.name for tree in original_trees for leaf in tree.get_leaves()})
    return taxa
