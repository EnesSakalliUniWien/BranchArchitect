"""Align child order while preserving non-moving taxa as anchors."""

from __future__ import annotations

from typing import List, Optional

from brancharchitect.tree import Node


def align_to_source_order(
    tree: Node,
    source_order: List[str],
    moving_taxa: Optional[set[str]] = None,
) -> None:
    """
    Align tree ordering to source_order, weighting non-moving taxa strongly.
    """
    if moving_taxa is None:
        moving_taxa = set()

    order_index = {name: i for i, name in enumerate(source_order)}
    n = len(source_order)

    def sort_key_for_leaf_names(leaf_names: List[str]) -> tuple[float, int]:
        if not leaf_names:
            return (float("inf"), n)
        total_weight = 0.0
        weighted_sum = 0.0
        min_non_mover_idx = n

        for leaf_name in leaf_names:
            idx = order_index.get(leaf_name, n)
            if leaf_name in moving_taxa:
                weight = 1.0
            else:
                weight = 100.0
                min_non_mover_idx = min(min_non_mover_idx, idx)

            weighted_sum += idx * weight
            total_weight += weight

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else float("inf")
        if min_non_mover_idx == n:
            min_non_mover_idx = order_index.get(leaf_names[0], n)

        return (weighted_avg, min_non_mover_idx)

    def reorder_node(node: Node) -> tuple[bool, List[str], tuple[float, int]]:
        if not node.children:
            leaf_names = [node.name]
            return False, leaf_names, sort_key_for_leaf_names(leaf_names)

        changed = False
        child_data: List[tuple[Node, List[str], tuple[float, int]]] = []
        for child in node.children:
            child_changed, child_leaf_names, child_sort_key = reorder_node(child)
            changed = child_changed or changed
            child_data.append((child, child_leaf_names, child_sort_key))

        sorted_child_data = sorted(child_data, key=lambda item: item[2])
        sorted_children = [child for child, _leaf_names, _sort_key in sorted_child_data]

        if sorted_children != node.children:
            node.children = sorted_children
            changed = True

        leaf_names: List[str] = []
        for _child, child_leaf_names, _sort_key in sorted_child_data:
            leaf_names.extend(child_leaf_names)

        return changed, leaf_names, sort_key_for_leaf_names(leaf_names)

    changed, _leaf_names, _sort_key = reorder_node(tree)
    if changed:
        tree.invalidate_caches(propagate_up=True)
