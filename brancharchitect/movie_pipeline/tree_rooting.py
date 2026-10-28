"""
Tree rooting utilities for phylogenetic trees.

This module provides functions for applying midpoint rooting to phylogenetic trees
to ensure consistent orientation for interpolation and visualization.
"""

from typing import Any, Dict, List, Tuple
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

    # Biopython only sees safe leaf ids, no internal labels and no metadata:
    # unquoted names with spaces or punctuation do not survive its parser, and
    # its node-based rerooting moves labels onto other branches. Names and
    # labels are restored by bipartition afterwards.
    safe_names = {name: f"t{index}" for name, index in original_encoding.items()}
    safe_encoding = {safe_names[name]: index for name, index in original_encoding.items()}

    rooted_newick_strings: List[str] = []
    for tree in trees:
        # 1. Convert brancharchitect.tree.Node to Newick string
        plain = tree.deep_copy(build_split_index=False)
        for node in plain.traverse():
            node.name = safe_names[node.name] if node.is_leaf() else ""
            node.values = {}
        newick_string = plain.to_newick()

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
    rooted_trees = parse_newick(
        "\n".join(rooted_newick_strings),
        order=[safe_names[name] for name in original_order],
        encoding=safe_encoding,
        force_list=True,
        treat_zero_as_epsilon=True,
    )

    real_names = {safe: name for name, safe in safe_names.items()}
    full_mask = (1 << len(original_encoding)) - 1
    for original, rooted in zip(trees, rooted_trees):
        _restore_names_and_labels(
            original, rooted, real_names, original_encoding, full_mask
        )

    return rooted_trees


def _unrooted_key(node: Node, full_mask: int) -> int:
    """The bipartition a branch represents, the same whichever side is the root."""
    mask = node.split_indices.bitmask
    return min(mask, full_mask ^ mask)


def _restore_names_and_labels(
    original: Node,
    rooted: Node,
    real_names: Dict[str, str],
    encoding: Dict[str, int],
    full_mask: int,
) -> None:
    """Put taxon names, internal labels and values back onto the rooted tree.

    Each label goes to the branch with the same bipartition, so support stays
    on the split it measures. The two new root branches share one
    bipartition; its label goes to the first of them only.
    """
    labels: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for node in original.traverse():
        if node is original:
            continue
        key = _unrooted_key(node, full_mask)
        # A rooted input also has two root branches on one bipartition.
        if key not in labels or not any(labels[key]):
            labels[key] = (node.name if node.children else "", dict(node.values))

    for leaf in rooted.get_leaves():
        leaf.name = real_names[leaf.name]
    rooted.invalidate_caches(propagate_up=False, propagate_down=True)
    rooted.initialize_split_indices(encoding)
    rooted.values = dict(original.values)

    restored: set[int] = set()
    for node in rooted.traverse():
        if node is rooted:
            continue
        key = _unrooted_key(node, full_mask)
        if key in restored or key not in labels:
            continue
        name, values = labels[key]
        if node.children:
            node.name = name
        node.values = dict(values)
        restored.add(key)
