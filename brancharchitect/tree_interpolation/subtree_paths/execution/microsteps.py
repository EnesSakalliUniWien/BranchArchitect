"""
Helper and processing functions for tree interpolation.

This module contains utility functions for processing split data,
extracting data, and managing the interpolation workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.consensus_tree.consensus_tree import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.consensus_tree.intermediate_tree import (
    create_subtree_grafted_tree,
    calculate_intermediate_implicit,
)
from .reordering import (
    apply_partial_reordering,
    create_reordering_strategy,  # type: ignore
    PartialOrderingStrategy,
)

# Type alias for the reordering strategy function
ReorderingStrategyFunc = Callable[..., PartialOrderingStrategy]


def _create_reordering_strategy_safe(
    strategy: str = "adaptive", distance_threshold: float = 0.2
) -> PartialOrderingStrategy:
    """Wrapper for create_reordering_strategy with proper typing."""
    return create_reordering_strategy(strategy, distance_threshold=distance_threshold)


logger: logging.Logger = logging.getLogger(__name__)


def extract_filtered_paths(
    selection: Dict[str, Any],
    active_changing_edge: Partition,
    subtree_partition: Partition,
) -> Tuple[List[Partition], List[Partition]]:
    """Extract and filter expand/collapse paths from selection, excluding specified partitions.

    Args:
        selection: Dictionary containing expand/collapse path segments
        active_changing_edge: Partition to exclude from paths
        subtree_partition: Subtree partition to exclude from paths

    Returns:
        Tuple of (create_path, collapse_path) with exclusions filtered out
    """
    exclusions = {active_changing_edge, subtree_partition}

    # Extract path segments - now guaranteed to be lists from builder
    expand_segments = selection.get("expand", {}).get("path_segment", []) or []
    collapse_segments = selection.get("collapse", {}).get("path_segment", []) or []

    expand_path: List[Partition] = [p for p in expand_segments if p not in exclusions]

    collapse_path: List[Partition] = [
        p for p in collapse_segments if p not in exclusions
    ]

    return expand_path, collapse_path


def apply_reference_weights_to_path(
    tree: Node,
    expand_path: List[Partition],
    reference_weights: Dict[Partition, float],
) -> None:
    """Set branch lengths on nodes along a path to match reference weights.

    Mutates the provided tree in place.
    """
    for ref_split in expand_path:
        node: Node | None = tree.find_node_by_split(ref_split)
        if node is not None:
            node.length = reference_weights.get(ref_split, 1)


def build_microsteps_for_selection(
    interpolation_state: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    active_changing_edge: Partition,
    selection: Dict[str, Any],
    tree_index: int,
    active_changing_edge_ordinal: int,
    step_idx: int,
) -> Tuple[List[Node], List[Optional[Partition]], Node]:
    """
    Build the 5 microsteps for a single selection under an active-changing edge.

    Steps:
    - IT_down: collapse zeros inside the subtree selection
    - C: collapse zero-length branches to consensus
    - C_reorder: partially reorder to match the reference
    - IT_up: graft the reference path while preserving order
    - IT_ref: apply final reference weights on the grafted path
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []

    def add_step(tree: Node, edge: Optional[Partition], subtree: Partition) -> None:
        # subtree still used internally for computations above; not tracked anymore
        trees.append(tree)
        edges.append(edge)

    # Extract and filter path segments using the modularized function
    subtree_partition = selection["subtree"]

    expand_path, collapse_path = extract_filtered_paths(
        selection, active_changing_edge, subtree_partition
    )

    it_down: Node = calculate_intermediate_implicit(
        interpolation_state, PartitionSet(set(collapse_path))
    )

    add_step(
        it_down,
        active_changing_edge,
        subtree_partition,
    )

    collapsed: Node = create_collapsed_consensus_tree(it_down, active_changing_edge)
    add_step(
        collapsed,
        active_changing_edge,
        subtree_partition,
    )

    # Apply partial reordering based on the interpolation context
    reordering_strategy = _create_reordering_strategy_safe(
        "adaptive", distance_threshold=0.2
    )

    reordered: Node = apply_partial_reordering(
        tree=collapsed,
        reference_tree=reference_tree,
        active_changing_edge=active_changing_edge,
        create_path=expand_path,
        collapse_path=collapse_path,
        strategy=reordering_strategy,
    )

    add_step(
        reordered,
        active_changing_edge,
        subtree_partition,
    )

    # Graft with order preservation to eliminate second reordering step
    pre_snap_reordered: Node = create_subtree_grafted_tree(
        base_tree=reordered,
        ref_path_to_build=expand_path,
        reference_tree=reference_tree,
        target_ordering_edge=active_changing_edge,
    )

    snapped_tree: Node = pre_snap_reordered.deep_copy()

    apply_reference_weights_to_path(
        snapped_tree,
        expand_path,
        reference_weights,
    )

    add_step(
        pre_snap_reordered,
        active_changing_edge,
        subtree_partition,
    )

    add_step(
        snapped_tree,
        active_changing_edge,
        subtree_partition,
    )

    return trees, edges, snapped_tree
