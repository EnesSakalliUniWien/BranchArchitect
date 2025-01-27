from typing import List, Dict, Optional
from brancharchitect.tree import Node  # Ensure this is the correct import


def compute_tree_imbalance(trees: List[Node]) -> Dict[str, float]:
    """
    Compute the imbalance of each tree, which can highlight asymmetries in the tree structures.

    Args:
    - trees (List[Node]): List of tree roots.

    Returns:
    - Dict[str, float]: A dictionary mapping tree identifiers to their imbalance scores.
    """
    imbalance_scores = {}
    for idx, tree in enumerate(trees):
        imbalance_scores[f"Tree_{idx}"] = _compute_imbalance(tree)
    return imbalance_scores


def _compute_imbalance(node: Node) -> float:
    """
    Recursively compute imbalance for a node.

    Args:
    - node (Node): Current node in the tree.

    Returns:
    - float: Imbalance score of the subtree rooted at this node.
    """
    if not node.children:
        return 0
    left = node.children[0]
    right = node.children[1] if len(node.children) > 1 else None
    left_size = _subtree_size(left)
    right_size = _subtree_size(right) if right else 0
    imbalance = abs(left_size - right_size)
    return (
        imbalance
        + _compute_imbalance(left)
        + (_compute_imbalance(right) if right else 0)
    )


def _subtree_size(node: Optional[Node]) -> int:
    """
    Compute the size of the subtree rooted at a node.

    Args:
    - node (Optional[Node]): The root node of the subtree.

    Returns:
    - int: Size of the subtree.
    """
    if not node:
        return 0
    size = 1
    for child in node.children:
        size += _subtree_size(child)
    return size


def annotate_tree_with_imbalance(tree: Node) -> None:
    """
    Annotate each node in the tree with its imbalance score.

    Args:
    - tree (Node): The root of the tree.
    """
    tree.values["imbalance"] = _compute_imbalance(tree)
    for child in tree.children:
        annotate_tree_with_imbalance(child)
