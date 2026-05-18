"""Build overlap-connected groups of mover expand paths."""

from __future__ import annotations

from typing import AbstractSet, Callable, Dict, Mapping, Set, Tuple

from brancharchitect.elements.partition import Partition


def form_overlap_path_groups(
    expand_paths: Mapping[Partition, AbstractSet[Partition]],
    overlap_graph: Mapping[Partition, Set[Partition]],
    order_key_for_group_member: Callable[[Partition], Tuple[int, ...]],
) -> tuple[list[set[Partition]], dict[Partition, int]]:
    """Form connected components of mover paths using overlap relationships."""
    if not expand_paths:
        return [], {}

    parent: Dict[Partition, Partition] = {subtree: subtree for subtree in expand_paths}
    rank: Dict[Partition, int] = {subtree: 0 for subtree in expand_paths}

    def find(subtree: Partition) -> Partition:
        if parent[subtree] != subtree:
            parent[subtree] = find(parent[subtree])
        return parent[subtree]

    def union(left: Partition, right: Partition) -> None:
        left_parent = find(left)
        right_parent = find(right)
        if left_parent == right_parent:
            return
        if rank[left_parent] < rank[right_parent]:
            left_parent, right_parent = right_parent, left_parent
        parent[right_parent] = left_parent
        if rank[left_parent] == rank[right_parent]:
            rank[left_parent] += 1

    for subtree, neighbors in overlap_graph.items():
        for neighbor in neighbors:
            union(subtree, neighbor)

    groups_by_root: Dict[Partition, Set[Partition]] = {}
    for subtree in expand_paths:
        root = find(subtree)
        groups_by_root.setdefault(root, set()).add(subtree)

    groups = sorted(
        groups_by_root.values(),
        key=lambda group: (
            min(len(expand_paths.get(subtree, set())) for subtree in group),
            min(order_key_for_group_member(subtree) for subtree in group),
        ),
    )

    subtree_to_group: dict[Partition, int] = {}
    for index, group in enumerate(groups):
        for subtree in group:
            subtree_to_group[subtree] = index

    return groups, subtree_to_group
