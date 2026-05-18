"""Detect relationships between destination expand paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Set, Tuple

from brancharchitect.elements.partition import Partition


@dataclass(slots=True)
class ExpandPathRelationships:
    """Pairwise overlap and containment relationships between mover paths."""

    overlap_graph: Dict[Partition, Set[Partition]]
    containment_edges: Set[Tuple[Partition, Partition]]
    successors: Dict[Partition, List[Partition]]


def build_expand_path_relationships(
    expand_paths: Mapping[Partition, AbstractSet[Partition]],
) -> ExpandPathRelationships:
    """Compute overlap and containment graphs for mover expand paths."""
    subtrees = list(expand_paths.keys())
    paths = {subtree: frozenset(expand_paths[subtree]) for subtree in subtrees}
    overlap_graph: Dict[Partition, Set[Partition]] = {}
    containment_edges: Set[Tuple[Partition, Partition]] = set()
    successors: Dict[Partition, List[Partition]] = {}

    for subtree in subtrees:
        overlap_graph[subtree] = set()
        successors[subtree] = []

    for index, subtree_a in enumerate(subtrees):
        path_a = paths[subtree_a]

        for subtree_b in subtrees[index + 1 :]:
            path_b = paths[subtree_b]

            if not path_a.isdisjoint(path_b):
                overlap_graph[subtree_a].add(subtree_b)
                overlap_graph[subtree_b].add(subtree_a)

            if path_a and path_b:
                if path_a < path_b:
                    containment_edges.add((subtree_a, subtree_b))
                    successors[subtree_a].append(subtree_b)
                elif path_b < path_a:
                    containment_edges.add((subtree_b, subtree_a))
                    successors[subtree_b].append(subtree_a)

    return ExpandPathRelationships(
        overlap_graph=overlap_graph,
        containment_edges=containment_edges,
        successors=successors,
    )
