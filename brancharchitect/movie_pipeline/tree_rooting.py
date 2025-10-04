"""
Tree rooting utilities for phylogenetic trees.

This module provides functions for applying midpoint rooting to phylogenetic trees
to ensure consistent orientation for interpolation and visualization.
"""

from typing import List
import io
from brancharchitect.tree import Node
from skbio import TreeNode as SkbioTreeNode
from brancharchitect.parser.newick_parser import parse_newick


def root_trees(trees: List[Node]) -> List[Node]:
    """
    Apply midpoint rooting to all trees for consistent orientation.

    Midpoint rooting places the root at the midpoint of the longest path
    between any two leaves, providing a consistent tree orientation that
    improves interpolation quality and visualization.

    This implementation uses scikit-bio for the rooting calculation.
    It converts each tree to the Newick format, reads it into a
    scikit-bio TreeNode, performs the rooting, and then converts it
    back to a brancharchitect Node object.

    Args:
        trees: List of trees to root

    Returns:
        List of midpoint-rooted trees (new copies, originals unchanged)
    """
    rooted_trees: List[Node] = []
    for tree in trees:
        # 1. Convert brancharchitect.tree.Node to Newick string
        newick_string = tree.to_newick()

        # 2. Create a skbio.TreeNode from the Newick string
        # Use a file-like wrapper for efficiency and clarity
        skbio_tree = SkbioTreeNode.read(io.StringIO(newick_string))

        # 3. Root the skbio.TreeNode at the midpoint
        rooted_skbio_tree = skbio_tree.root_at_midpoint()

        # 4. Convert the rooted skbio.TreeNode back to a Newick string
        rooted_newick_string = str(rooted_skbio_tree)

        # 5. Parse the new Newick string back to a brancharchitect.tree.Node
        # The parser returns a list, so we take the first element.
        rooted_tree = parse_newick(
            rooted_newick_string, force_list=True, treat_zero_as_epsilon=True
        )[0]
        rooted_trees.append(rooted_tree)

    return rooted_trees
