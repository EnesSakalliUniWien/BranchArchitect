from __future__ import annotations
import logging
from typing import Dict, List
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

logger: logging.Logger = logging.getLogger(__name__)


def calculate_consensus_tree(tree: Node, split_dict: Dict[Partition, float]) -> Node:
    """Calculate consensus tree based on split dictionary."""
    consensus_tree: Node = tree.deep_copy()
    return _calculate_consensus_tree(consensus_tree, split_dict)


def collapse_zero_length_branches_for_node(
    node: Node, tol: float = 0.0, destination_tree: Node | None = None
) -> None:
    """
    Collapse branches whose length is effectively zero (<= tol) in `node`'s subtree.
    Runs to a fixpoint and refreshes indices/caches after structural changes.

    If destination_tree is provided, only collapse branches whose splits DON'T exist
    in the destination tree. This ensures we preserve topology needed for the final tree.
    """
    # Pre-compute destination splits if provided
    destination_splits = None
    if destination_tree is not None:
        destination_splits = set(destination_tree.to_splits())

    def _collapse_once(cur: Node) -> bool:
        changed = False
        if not cur.children:
            return False

        new_children: List[Node] = []
        for ch in cur.children:
            ch_len = 0.0 if ch.length is None else float(ch.length)
            if ch_len <= tol and ch.children:
                # Check if this split should be preserved
                should_collapse = True
                if destination_splits is not None:
                    if ch.split_indices in destination_splits:
                        # Split exists in destination - DON'T collapse
                        should_collapse = False
                        logger.debug(
                            f"[CONSENSUS COLLAPSE] Preserving zero-length branch "
                            f"(split exists in destination): {ch.split_indices}"
                        )

                if should_collapse:
                    # splice grandchildren
                    for g in ch.children:
                        g.parent = cur
                        new_children.append(g)
                    changed = True
                else:
                    # Keep the branch even though it's zero-length
                    new_children.append(ch)
            else:
                new_children.append(ch)

        if changed:
            cur.children = new_children

        # Recurse into children
        for ch in cur.children:
            changed = _collapse_once(ch) or changed
        return changed

    # Fixpoint: keep collapsing until no structural change occurs
    while _collapse_once(node):
        pass

    # Rebuild split indices & caches once after topology edits
    root = node.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches(propagate_up=True)


def _calculate_consensus_tree(node: Node, split_dict: Dict[Partition, float]) -> Node:
    """Recursively build consensus tree by keeping only supported splits."""
    if not node.children:
        return node

    new_children: List[Node] = []
    for child in node.children:
        processed_child = _calculate_consensus_tree(child, split_dict)
        if processed_child.children:
            if processed_child.split_indices in split_dict:
                new_children.append(processed_child)
            else:
                for grandchild in processed_child.children:
                    grandchild.parent = node
                    new_children.append(grandchild)
        else:
            new_children.append(processed_child)

    node.children = new_children
    for child in node.children:
        child.parent = node
    return node


def reorder_consensus_tree_by_edge(
    consensus_tree: Node,
    target_tree: Node,
    edge: Partition,
) -> Node:
    """
    Reorder consensus_tree's node at 'edge' to match the leaf order from target_tree's corresponding node.
    """
    reordered_tree = consensus_tree.deep_copy()

    target_node = target_tree.find_node_by_split(edge)
    node_to_reorder = reordered_tree.find_node_by_split(edge)

    if target_node and node_to_reorder:
        correct_leaf_order = list(target_node.get_current_order())
        try:
            node_to_reorder.reorder_taxa(correct_leaf_order)
        except ValueError as e:
            logger.warning(
                f"Could not reorder node for pivot edge {edge} due to taxa mismatch: {e}. "
                "The order may be partially inconsistent for this step."
            )

    return reordered_tree


def create_collapsed_consensus_tree(
    down_phase_tree: Node,
    pivot_edge: Partition,
    tol: float = 0,  # small tolerance for numeric stability
    destination_tree: Node
    | None = None,  # to preserve splits that exist in destination
) -> Node:
    """
    Copy down_phase_tree, collapse zero-length internal branches under pivot_edge, and refresh indices.

    If destination_tree is provided, only collapse zero-length branches whose splits DON'T
    exist in the destination. This ensures the final tree topology matches the destination.
    """
    collapsed_tree: Node = down_phase_tree.deep_copy()
    consensus_edge_node: Node | None = collapsed_tree.find_node_by_split(pivot_edge)
    if consensus_edge_node is not None:
        collapse_zero_length_branches_for_node(
            consensus_edge_node, tol=tol, destination_tree=destination_tree
        )

    root: Node = collapsed_tree.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches(propagate_up=True)
    return collapsed_tree
