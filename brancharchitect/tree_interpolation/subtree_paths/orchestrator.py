"""Sequence orchestrator for subtree-path interpolation.

Coordinates the high-level interpolation sequence for each active-changing
split by:
- Calculating per-split subtree paths
- Planning collapse/expand steps per subtree
- Executing the 5 microsteps and aggregating results
"""

from __future__ import annotations

from typing import Dict, List, Optional
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .paths import calculate_subtree_paths
from .execution.step_executor import apply_stepwise_plan_for_edge


def create_interpolation_for_active_split_sequence(
    source_tree: Node,
    destination_tree: Node,
    target_active_changing_edges: List[Partition],
    jumping_subtree_solutions: Dict[Partition, List[List[Partition]]],
) -> tuple[
    List[Node],
    List[Partition],
    List[Optional[Partition]],
]:
    """
    Create an interpolation sequence from source to destination tree for active-changing splits.

    Strategy (modular and stepwise):
      1) Calculate subtree paths once for destination and source.
      2) For each active_changing_edge on the current interpolation state:
         - If the active_changing_edge is missing in either tree, run classical fallback.
         - Else, try stepwise plan:
             a) Iterate selections (individual/whole per rule).
             b) Apply micro-steps per selection.
           If no selections are produced, run the simple 5-step fallback for that active_changing_edge.
    """
    interpolation_sequence: List[Node] = []
    failed_active_changing_edges: List[Partition] = []
    processed_active_changing_edge_tracking: List[Optional[Partition]] = []
    # subtree tracking removed

    interpolation_state: Node = source_tree.deep_copy()

    # Precompute subtree paths in destination/source
    destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
        jumping_subtree_solutions, destination_tree, source_tree
    )

    current_base_tree = interpolation_state.deep_copy()

    for _, active_changing_edge in enumerate(target_active_changing_edges):
        current_base_tree: Node = interpolation_state.deep_copy()

        current_base_tree.initialize_split_indices(current_base_tree.taxa_encoding)

        # Paths for this active_changing_edge (kept separate for destination/source)
        # Keep as PartitionSet[Partition] - no conversion needed
        source_paths_for_s_edge = source_subtree_paths.get(active_changing_edge, {})
        destination_paths_for_s_edge = destination_subtree_paths.get(active_changing_edge, {})

        step_trees, step_edges, new_state = apply_stepwise_plan_for_edge(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            active_changing_edge=active_changing_edge,
            expand_paths_for_s_edge=destination_paths_for_s_edge,
            collapse_paths_for_s_edge=source_paths_for_s_edge,
        )

        if step_trees:
            interpolation_sequence.extend(step_trees)
            processed_active_changing_edge_tracking.extend(step_edges)
            interpolation_state = new_state

    return (
        interpolation_sequence,
        failed_active_changing_edges,
        processed_active_changing_edge_tracking,
    )
