"""Cycle detection for expand-path containment dependencies."""

from __future__ import annotations

from typing import AbstractSet, List, Mapping, Optional, Set

from brancharchitect.elements.partition import Partition


def find_containment_cycle(
    expand_paths: Mapping[Partition, AbstractSet[Partition]],
    successors: Mapping[Partition, list[Partition]],
) -> Optional[List[Partition]]:
    """Return a containment cycle if one exists."""
    visited: Set[Partition] = set()
    recursion_stack: Set[Partition] = set()

    def visit(node: Partition, path: List[Partition]) -> Optional[List[Partition]]:
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in successors.get(node, []):
            if neighbor in recursion_stack:
                cycle_start = path.index(neighbor) if neighbor in path else 0
                return path[cycle_start:] + [neighbor]
            if neighbor not in visited:
                result = visit(neighbor, path + [neighbor])
                if result:
                    return result

        recursion_stack.remove(node)
        return None

    for subtree in expand_paths:
        if subtree not in visited:
            cycle = visit(subtree, [subtree])
            if cycle:
                return cycle

    return None
