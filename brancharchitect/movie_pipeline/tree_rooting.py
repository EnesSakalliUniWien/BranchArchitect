"""
Tree rooting utilities for phylogenetic trees.

This module provides functions for applying midpoint rooting to phylogenetic trees
to ensure consistent orientation for interpolation and visualization.
"""

from typing import List
import io
from Bio.Phylo.NewickIO import Parser, write
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick


def root_trees(trees: List[Node]) -> List[Node]:
    """
    Apply midpoint rooting to all trees for consistent orientation.

    Midpoint rooting places the root at the midpoint of the longest path
    between any two leaves, providing a consistent tree orientation that
    improves interpolation quality and visualization.

    This implementation uses Biopython for the rooting calculation.
    It converts each tree to the Newick format, reads it into a
    Biopython tree, performs the rooting, and then converts it
    back to a brancharchitect Node object.

    Args:
        trees: List of trees to root

    Returns:
        List of midpoint-rooted trees (new copies, originals unchanged)
    """
    # Preserve original taxa encoding to ensure consistent split indices after rooting
    original_encoding = trees[0].taxa_encoding
    original_order = list(trees[0].get_current_order())

    rooted_newick_strings: List[str] = []
    for tree in trees:
        # 1. Convert brancharchitect.tree.Node to Newick string
        newick_string = tree.to_newick()

        # 2. Create a Biopython tree from the Newick string
        bio_tree = next(Parser.from_string(newick_string).parse())

        # 3. Root the tree at the midpoint
        bio_tree.root_at_midpoint()

        # 4. Convert the rooted tree back to a Newick string
        output = io.StringIO()
        write([bio_tree], output)
        rooted_newick_string = output.getvalue().strip()

        rooted_newick_strings.append(rooted_newick_string)

    # 5. Parse the new Newick strings back to brancharchitect.tree.Node objects
    # CRITICAL: Pass original order/encoding to preserve consistent split indices
    rooted_trees: List[Node] = parse_newick(  # type: ignore[assignment]
        "\n".join(rooted_newick_strings),
        order=original_order,
        encoding=original_encoding,
        force_list=True,
        treat_zero_as_epsilon=True,
    )

    return rooted_trees
