from typing import Optional, Tuple, List
from brancharchitect.tree import Node


def fallback_to_simple_rerooting(tree1: Node, tree2: Node) -> Node:
    """
    Fallback to simple rerooting strategy when advanced methods fail.

    Args:
        tree1: Tree to reroot
        tree2: Reference tree

    Returns:
        Rerooted tree using simple strategy
    """
    from .core_rooting import simple_reroot

    return simple_reroot(tree1, tree2)


def select_best_non_leaf_candidate(
    candidates: List[Tuple[Node, float]],
) -> Optional[Node]:
    """
    Select the best non-leaf candidate from the candidate list.

    Args:
        candidates: List of (node, score) candidates

    Returns:
        Best non-leaf candidate node, or None if none found
    """
    for node, _score in candidates:
        if node.children:  # Non-leaf node
            return node
    return None


def validate_and_rebuild_tree_structure(rerooted_tree: Node) -> Node:
    """
    Validate and rebuild tree structure after rerooting.

    Args:
        rerooted_tree: The rerooted tree to validate

    Returns:
        Validated and possibly rebuilt tree
    """
    # Basic validation - ensure root has no parent
    if rerooted_tree.parent is not None:
        rerooted_tree.parent = None

    # Additional structural validation could be added here
    return rerooted_tree
