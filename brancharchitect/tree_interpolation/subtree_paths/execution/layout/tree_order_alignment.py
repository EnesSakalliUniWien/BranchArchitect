"""Align child order while preserving non-moving taxa as anchors."""

from __future__ import annotations

from typing import List, Optional, Tuple

from brancharchitect.tree import Node


_OrderAggregate = Tuple[float, float, int, int]


def _sort_key(aggregate: _OrderAggregate, fallback_idx: int) -> tuple[float, int]:
    weighted_sum, total_weight, min_non_mover_idx, first_leaf_idx = aggregate
    weighted_avg = weighted_sum / total_weight if total_weight > 0 else float("inf")
    min_idx = min_non_mover_idx if min_non_mover_idx != fallback_idx else first_leaf_idx
    return (weighted_avg, min_idx)


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

    def leaf_aggregate(leaf_name: str) -> _OrderAggregate:
        idx = order_index.get(leaf_name, n)
        if leaf_name in moving_taxa:
            return (idx, 1.0, n, idx)
        return (idx * 100.0, 100.0, idx, idx)

    def combine_aggregates(
        sorted_child_data: List[tuple[Node, _OrderAggregate, tuple[float, int]]],
    ) -> _OrderAggregate:
        weighted_sum = 0.0
        total_weight = 0.0
        min_non_mover_idx = n
        first_leaf_idx = sorted_child_data[0][1][3] if sorted_child_data else n

        for _child, aggregate, _key in sorted_child_data:
            child_weighted_sum, child_total_weight, child_min_idx, _first_idx = (
                aggregate
            )
            weighted_sum += child_weighted_sum
            total_weight += child_total_weight
            if child_min_idx < min_non_mover_idx:
                min_non_mover_idx = child_min_idx

        return (weighted_sum, total_weight, min_non_mover_idx, first_leaf_idx)

    # Children before parents (reversed pre-order), not recursion, so
    # ladder-shaped trees deeper than the recursion limit still align.
    results: dict[int, tuple[_OrderAggregate, tuple[float, int]]] = {}
    changed = False
    for node in reversed(tree.traverse()):
        if not node.children:
            aggregate = leaf_aggregate(str(node.name))
            results[id(node)] = (aggregate, _sort_key(aggregate, n))
            continue

        child_data: List[tuple[Node, _OrderAggregate, tuple[float, int]]] = [
            (child, *results[id(child)]) for child in node.children
        ]
        sorted_child_data = sorted(child_data, key=lambda item: item[2])
        sorted_children = [child for child, _aggregate, _child_key in sorted_child_data]

        if sorted_children != node.children:
            node.children = sorted_children
            changed = True

        aggregate = combine_aggregates(sorted_child_data)
        results[id(node)] = (aggregate, _sort_key(aggregate, n))

    if changed:
        tree.invalidate_caches(propagate_up=True)
