from typing import List

from brancharchitect.tree import Node


def get_taxa_circular_order(node: Node) -> List[str]:
    """
    Compute the circular order of leaf names in a tree.

    The leaf-order distance tests use this as the bridge between tree topology
    and the circular distance helpers.
    """
    taxa_order: List[str] = []
    _get_taxa_circular_order(node, taxa_order)
    return taxa_order


def _get_taxa_circular_order(node: Node, taxa_order: List[str]) -> None:
    """Append leaf names from a preorder traversal into ``taxa_order``."""
    if not node.children:
        taxa_order.append(node.name)
        return
    for child in node.children:
        _get_taxa_circular_order(child, taxa_order)
