"""
Edge Depth Ordering

This module provides hierarchical ordering for lattice edges based on subset
relationships and tree depth to ensure optimal processing order.

Key principle: Process subsets before supersets
    Example: {A} → {B} → {A,B} → {A,B,C}

This incremental ordering is critical for the lattice algorithm's approach:
resolving simple conflicts first may simplify or eliminate larger conflicts.

Functions:
    - topological_sort_edges: Sort partitions topologically (subsets before supersets)
    - sort_pivot_edges_by_subset_hierarchy: Sort PivotEdgeSubproblem objects for optimal processing
"""

from __future__ import annotations
import heapq
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import is_full_overlap
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.logger import jt_logger


def _compute_depths(edges: List[Partition], tree: Node) -> Dict[Partition, float]:
    """Pre-compute tree depths for all edges."""
    depths: Dict[Partition, float] = {}
    for edge in edges:
        node = tree.find_node_by_split(edge)
        depths[edge] = node.depth if node and node.depth is not None else 0
    return depths


def _build_dependency_graph(
    edges: List[Partition],
) -> Tuple[Dict[Partition, List[Partition]], Dict[Partition, int]]:
    """
    Build adjacency list and in-degree map for subset relationships.

    adj[u] = [v, ...] means u is a subset of v (u -> v).
    """
    adj: Dict[Partition, List[Partition]] = {u: [] for u in edges}
    in_degree: Dict[Partition, int] = dict.fromkeys(edges, 0)

    for u in edges:
        for v in edges:
            if u != v and is_full_overlap(u, v):
                adj[u].append(v)
                in_degree[v] += 1

    return adj, in_degree


def _run_kahn_algorithm(
    edges: List[Partition],
    adj: Dict[Partition, List[Partition]],
    in_degree: Dict[Partition, int],
    depths: Dict[Partition, float],
) -> List[Partition]:
    """
    Run Kahn's algorithm with priority queue for topological sort.

    Priority Key: (size, tree_depth, indices) - smaller values first.
    """
    queue: List[Tuple[int, float, Tuple[int, ...], Partition]] = []

    for u in edges:
        if in_degree[u] == 0:
            heapq.heappush(queue, (len(u), depths[u], u.indices, u))

    sorted_result: List[Partition] = []

    while queue:
        _, _, _, u = heapq.heappop(queue)
        sorted_result.append(u)

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(queue, (len(v), depths[v], v.indices, v))

    return sorted_result


def topological_sort_edges(edges: List[Partition], tree: Node) -> List[Partition]:
    """
    Sort edges topologically based on subset relationships.

    Ensures that if A is a proper subset of B, A comes before B in the sorted list.
    Uses a Directed Acyclic Graph (DAG) where an edge A -> B exists if A is a subset of B.
    Kahn's algorithm is used with a priority queue to handle secondary sorting criteria.

    Secondary Sort Criteria (Priority Queue):
    1. Partition Size (smaller first)
    2. Tree Depth (shallower/leaves first)
    3. Taxa Indices (deterministic tie-breaker)

    Args:
        edges: List of partitions to sort
        tree: Tree to use for depth lookups (secondary criterion)

    Returns:
        List of partitions sorted topologically (Subsets -> Supersets)
    """
    if not edges:
        return []

    depths = _compute_depths(edges, tree)
    adj, in_degree = _build_dependency_graph(edges)
    sorted_result = _run_kahn_algorithm(edges, adj, in_degree, depths)

    # Check for cycles (should not happen with subset relation)
    if len(sorted_result) != len(edges):
        jt_logger.warning(
            f"Topological sort incomplete! Cycles detected? "
            f"Sorted {len(sorted_result)}/{len(edges)} edges."
        )
        remaining = [e for e in edges if e not in sorted_result]
        sorted_result.extend(remaining)

    return sorted_result


def sort_pivot_edges_by_subset_hierarchy(
    pivot_edges: List[PivotEdgeSubproblem],
    tree1: Node,
    _tree2: Node,
) -> List[PivotEdgeSubproblem]:
    """
    Sort pivot edges by subset hierarchy using tree depth for optimal processing order.

    Ensures that smaller, simpler conflicts are resolved before larger, more complex ones.
    This ordering is critical for the lattice algorithm's incremental approach:

    - Subsets processed before supersets (e.g., {A} → {A,B} → {A,B,C})
    - Tree depth provides fine-grained ordering within same subset level
    - Smaller partitions processed first, potentially simplifying larger conflicts

    Uses topological sort directly on tree1 (primary tree) for efficiency.

    Args:
        pivot_edges: List of PivotEdgeSubproblem objects to sort
        tree1: First phylogenetic tree for depth calculation (primary)
        _tree2: Second phylogenetic tree (reserved for future use)

    Returns:
        Sorted list of pivot edges in ascending order (subsets first)

    Example:
        Given edges for partitions {A}, {A,B}, {B}:
        Returns them ordered as: [{A}, {B}, {A,B}]
    """
    if not pivot_edges:
        return pivot_edges

    # Extract partitions and perform topological sort directly
    partitions = [edge.pivot_split for edge in pivot_edges]
    sorted_partitions = topological_sort_edges(partitions, tree1)

    # Build partition -> index map for O(1) lookup
    partition_order = {p: i for i, p in enumerate(sorted_partitions)}

    # Sort pivot edges by their topological order
    sorted_edges = sorted(pivot_edges, key=lambda e: partition_order[e.pivot_split])

    if not jt_logger.disabled:
        jt_logger.debug(
            f"Sorted {len(pivot_edges)} pivot edges by subset hierarchy and depth"
        )
        for i, edge in enumerate(sorted_edges):
            jt_logger.debug(f"  {i + 1}. {edge.pivot_split}")

    return sorted_edges
