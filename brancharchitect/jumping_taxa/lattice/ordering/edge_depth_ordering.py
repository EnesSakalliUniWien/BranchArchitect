"""
Edge Depth Ordering

This module provides depth computation and hierarchical ordering for lattice edges.
Orders edges based on subset relationships and tree depth to ensure optimal
processing: smaller, simpler conflicts are resolved before larger, complex ones.

Key principle: Process subsets before supersets
    Example: {A} → {B} → {A,B} → {A,B,C}

This incremental ordering is critical for the lattice algorithm's approach:
resolving simple conflicts first may simplify or eliminate larger conflicts.

Functions:
    - compute_pivot_edge_depths: Calculate depth values respecting hierarchy
    - sort_pivot_edges_by_subset_hierarchy: Sort edges for optimal processing
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
from brancharchitect.jumping_taxa.debug import jt_logger


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

    # 1. Build Dependency Graph
    # adj[u] = [v, ...] means u is a subset of v (u -> v)
    adj: Dict[Partition, List[Partition]] = {u: [] for u in edges}
    in_degree: Dict[Partition, int] = {u: 0 for u in edges}

    # Pre-compute depths for tie-breaking
    depths: Dict[Partition, float] = {}
    for edge in edges:
        node = tree.find_node_by_split(edge)
        depths[edge] = node.depth if node and node.depth is not None else 0

    # Build edges: O(N^2)
    for u in edges:
        for v in edges:
            if u == v:
                continue
            # If u is a subset of v, u must be processed before v.
            # So u -> v. v depends on u.
            if is_full_overlap(u, v):
                adj[u].append(v)
                in_degree[v] += 1

    # 2. Initialize Priority Queue (Min-Heap)
    # We want to process nodes with in-degree 0 (no unprocessed subsets).
    # Priority Key: (size, tree_depth, indices)
    queue: List[Tuple[int, float, Tuple[int, ...], Partition]] = []

    for u in edges:
        if in_degree[u] == 0:
            # Ensure indices are a comparable tuple
            # Partition.indices is already a sorted tuple of ints
            heapq.heappush(queue, (len(u), depths[u], u.indices, u))

    # 3. Process Queue
    sorted_result: List[Partition] = []

    while queue:
        _, _, _, u = heapq.heappop(queue)
        sorted_result.append(u)

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(queue, (len(v), depths[v], v.indices, v))

    # Check for cycles (should not happen with subset relation)
    if len(sorted_result) != len(edges):
        jt_logger.warning(
            f"Topological sort incomplete! Cycles detected? "
            f"Sorted {len(sorted_result)}/{len(edges)} edges."
        )
        # Fallback: append remaining edges (undefined order)
        remaining = [e for e in edges if e not in sorted_result]
        sorted_result.extend(remaining)

    return sorted_result


def compute_pivot_edge_depths(
    pivot_edges: List[Partition], target: Node
) -> Dict[Partition, float]:
    """
    Compute depths for pivot edges using explicit topological sort.

    Replaces the previous heuristic scoring with a rank based on topological order.

    Args:
        pivot_edges: List of partitions to compute depths for
        target: Tree node to use for base depth calculation

    Returns:
        Dictionary mapping each partition to its topological rank (0-based index)
    """
    # Perform topological sort
    sorted_edges = topological_sort_edges(pivot_edges, target)

    # Map edge to its rank
    return {edge: float(i) for i, edge in enumerate(sorted_edges)}


def sort_pivot_edges_by_subset_hierarchy(
    pivot_edges: List[PivotEdgeSubproblem], tree1: Node, tree2: Node
) -> List[PivotEdgeSubproblem]:
    """
    Sort pivot edges by subset hierarchy using tree depth for optimal processing order.

    Ensures that smaller, simpler conflicts are resolved before larger, more complex ones.
    This ordering is critical for the lattice algorithm's incremental approach:

    - Subsets processed before supersets (e.g., {A} → {A,B} → {A,B,C})
    - Tree depth provides fine-grained ordering within same subset level
    - Smaller partitions processed first, potentially simplifying larger conflicts

    The sorting uses average depth across both trees to balance the importance
    of each tree's structure in determining processing order.

    Args:
        pivot_edges: List of PivotEdgeSubproblem objects to sort
        tree1: First phylogenetic tree for depth calculation
        tree2: Second phylogenetic tree for depth calculation

    Returns:
        Sorted list of pivot edges in ascending order (subsets first)

    Example:
        Given edges for partitions {A}, {A,B}, {B}:
        Returns them ordered as: [{A}, {B}, {A,B}]
    """
    if not pivot_edges:
        return pivot_edges

    # Extract partitions from pivot edges
    partitions = [edge.pivot_split for edge in pivot_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_pivot_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_pivot_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths: Dict[Partition, float] = {}
    for partition in partitions:
        avg_depths[partition] = (depths1[partition] + depths2[partition]) / 2

    # Sort pivot edges by their average depths (ascending = subsets first)
    sorted_edges: List[PivotEdgeSubproblem] = sorted(
        pivot_edges,
        key=lambda edge: avg_depths[edge.pivot_split],
    )

    jt_logger.debug(
        f"Sorted {len(pivot_edges)} pivot edges by subset hierarchy and depth"
    )
    for i, edge in enumerate(sorted_edges):
        depth = avg_depths[edge.pivot_split]
        jt_logger.debug(f"  {i + 1}. {edge.pivot_split} (avg_depth={depth})")

    return sorted_edges
