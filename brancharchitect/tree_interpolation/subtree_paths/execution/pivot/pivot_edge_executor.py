from __future__ import annotations

from typing import Dict, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import SprMoveEvent

from ...planning import PivotTransitionPlan, build_pivot_transition_plan
from ..events import build_spr_move_event, highlight_group_for_event
from ..grouping import (
    compute_sibling_groups,
    get_collapse_splits,
    get_expand_splits,
)
from ..layout.mover_ordering import (
    build_destination_mover_order_key,
    sort_mover_partitions,
)
from ..phases import build_subtree_interpolation_frames


def execute_pivot_edge_interpolation(
    current_base_tree: Node,
    destination_tree: Node,
    source_tree: Node,
    current_pivot_edge: Partition,
    expand_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    collapse_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    source_weights: Optional[Dict[Partition, float]] = None,
    destination_weights: Optional[Dict[Partition, float]] = None,
) -> tuple[
    list[Node],
    list[Optional[Partition]],
    Node,
    list[list[Partition]],
    list[SprMoveEvent],
]:
    """Execute all mover microsteps for one pivot edge."""
    trees: list[Node] = []
    edges: list[Optional[Partition]] = []
    subtree_highlight_tracker: list[list[Partition]] = []
    spr_move_events: list[SprMoveEvent] = []
    interpolation_state: Node = current_base_tree
    expand_paths_for_plan = dict(expand_paths_for_pivot_edge)
    collapse_paths_for_plan = dict(collapse_paths_for_pivot_edge)
    initial_mover_partition_set = set(expand_paths_for_plan) | set(
        collapse_paths_for_plan
    )
    initial_subtree_order_key = build_destination_mover_order_key(
        destination_tree,
        current_pivot_edge,
        initial_mover_partition_set,
    )

    selections: PivotTransitionPlan = build_pivot_transition_plan(
        expand_paths_for_plan,
        collapse_paths_for_plan,
        source_tree,
        destination_tree,
        current_pivot_edge=current_pivot_edge,
        subtree_order_key=initial_subtree_order_key,
    )

    if source_weights is None:
        source_weights = source_tree.to_weighted_splits()
    if destination_weights is None:
        destination_weights = destination_tree.to_weighted_splits()

    mover_partition_set = (
        set(expand_paths_for_plan) | set(collapse_paths_for_plan) | set(selections)
    )
    subtree_order_key = build_destination_mover_order_key(
        destination_tree,
        current_pivot_edge,
        mover_partition_set,
    )
    all_mover_partitions = sort_mover_partitions(
        mover_partition_set,
        subtree_order_key,
    )

    collapse_splits = get_collapse_splits(collapse_paths_for_plan)
    expand_splits = get_expand_splits(expand_paths_for_plan)
    collapse_sibling_groups, expand_sibling_groups = compute_sibling_groups(
        all_mover_partitions,
        collapse_splits,
        expand_splits,
        source_parent_map,
        dest_parent_map,
    )

    for i, (subtree, selection) in enumerate(selections.items()):
        is_first_mover = i == 0

        step_start = len(trees)
        step_trees, step_edges, interpolation_state, step_highlight_tracker = (
            build_subtree_interpolation_frames(
                interpolation_state=interpolation_state,
                destination_tree=destination_tree,
                current_pivot_edge=current_pivot_edge,
                selection=selection,
                all_mover_partitions=all_mover_partitions,
                source_parent_map=source_parent_map,
                dest_parent_map=dest_parent_map,
                is_first_mover=is_first_mover,
                source_weights=source_weights,
                destination_weights=destination_weights,
                collapse_sibling_groups=collapse_sibling_groups,
                expand_sibling_groups=expand_sibling_groups,
            )
        )

        if step_trees:
            step_end = step_start + len(step_trees) - 1
            spr_move_events.append(
                build_spr_move_event(
                    current_pivot_edge=current_pivot_edge,
                    driver_subtree=subtree,
                    highlight_group=highlight_group_for_event(
                        subtree, step_highlight_tracker
                    ),
                    selection=selection,
                    source_weights=source_weights,
                    destination_weights=destination_weights,
                    step_range=(step_start, step_end),
                )
            )

        trees.extend(step_trees)
        edges.extend(step_edges)
        subtree_highlight_tracker.extend(step_highlight_tracker)

    if not trees:
        trees.append(interpolation_state.deep_copy())
        edges.append(current_pivot_edge)
        subtree_highlight_tracker.append([])

    return trees, edges, interpolation_state, subtree_highlight_tracker, spr_move_events
