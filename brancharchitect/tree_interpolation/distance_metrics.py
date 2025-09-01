"""
Distance metric calculations for tree interpolation.

This module provides functions for calculating various distance metrics
between jumping taxa components and s-edges in phylogenetic trees.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node


def calculate_component_distances(
    tree: Node, component: Partition, s_edge: Partition
) -> Tuple[float, float]:
    """Calculate topological and weighted distances between component and s-edge in a tree."""
    path = tree.find_path_between_splits(component, s_edge)
    if not path:
        return 0.0, 0.0

    # Topological distance: number of edges in path
    topological_dist = len(path) - 1

    # Weighted distance: sum of branch lengths in path
    weighted_dist = sum(
        node.length if node.length is not None else 0.0
        for node in path[1:]  # Skip first node (component node itself)
    )

    return float(topological_dist), weighted_dist


def process_solution_sets(
    target: Node,
    reference: Node,
    s_edge: Partition,
    solution_sets: List[List[Partition]],
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """Process all solution sets for an s-edge and collect distance data."""
    target_topological_distances: List[float] = []
    target_weighted_distances: List[float] = []
    reference_topological_distances: List[float] = []
    reference_weighted_distances: List[float] = []
    total_components = 0

    for solution_set in solution_sets:
        for component in solution_set:
            total_components += 1

            # Calculate distances in both trees
            target_topological, target_weighted = calculate_component_distances(
                target, component, s_edge
            )
            ref_topological, ref_weighted = calculate_component_distances(
                reference, component, s_edge
            )

            target_topological_distances.append(target_topological)
            target_weighted_distances.append(target_weighted)
            reference_topological_distances.append(ref_topological)
            reference_weighted_distances.append(ref_weighted)

    return (
        target_topological_distances,
        target_weighted_distances,
        reference_topological_distances,
        reference_weighted_distances,
        total_components,
    )


def calculate_s_edge_distances(
    target: Node,
    reference: Node,
    lattice_edge_solutions: Dict[Partition, List[List[Partition]]],
) -> Dict[Partition, Dict[str, float]]:
    """
    Calculate distances from jumping taxa components to their corresponding s-edges.

    Computes both topological and weighted distances from all jumping taxa components
    to the s-edge node in both target and reference trees.

    Args:
        target: Target tree for interpolation
        reference: Reference tree for interpolation
        lattice_edge_solutions: Dictionary mapping s-edges to their solution sets

    Returns:
        Dictionary mapping each s-edge to distance metrics:
        - "target_topological": Average topological distance in target tree
        - "target_weighted": Average weighted distance in target tree
        - "reference_topological": Average topological distance in reference tree
        - "reference_weighted": Average weighted distance in reference tree
        - "total_topological": Sum of target and reference topological distances
        - "total_weighted": Sum of target and reference weighted distances
        - "component_count": Number of jumping taxa for this s-edge
    """
    s_edge_distances: Dict[Partition, Dict[str, float]] = {}

    for s_edge, solution_sets in lattice_edge_solutions.items():
        (
            target_topological_distances,
            target_weighted_distances,
            reference_topological_distances,
            reference_weighted_distances,
            total_components,
        ) = process_solution_sets(target, reference, s_edge, solution_sets)

        # Calculate averages
        if total_components > 0:
            avg_target_topological = sum(target_topological_distances) / len(
                target_topological_distances
            )
            avg_target_weighted = sum(target_weighted_distances) / len(
                target_weighted_distances
            )
            avg_reference_topological = sum(reference_topological_distances) / len(
                reference_topological_distances
            )
            avg_reference_weighted = sum(reference_weighted_distances) / len(
                reference_weighted_distances
            )
        else:
            avg_target_topological = avg_target_weighted = 0.0
            avg_reference_topological = avg_reference_weighted = 0.0

        s_edge_distances[s_edge] = {
            "target_topological": avg_target_topological,
            "target_weighted": avg_target_weighted,
            "reference_topological": avg_reference_topological,
            "reference_weighted": avg_reference_weighted,
            "total_topological": avg_target_topological + avg_reference_topological,
            "total_weighted": avg_target_weighted + avg_reference_weighted,
            "component_count": float(total_components),
        }

    return s_edge_distances
