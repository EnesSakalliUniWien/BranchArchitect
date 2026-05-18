"""Build executable mover plans for one active pivot edge."""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Dict, Mapping, Optional, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node

from ...analysis.split_analysis import (
    find_incompatible_splits,
    get_unique_splits_for_current_pivot_edge_subtree,
)
from .path_assembly import (
    assemble_collapse_path,
    assemble_expand_path,
    gather_subtree_transition_splits,
    mark_transition_step_processed,
    store_ordered_transition_step,
)
from .transition_step import PivotTransitionPlan
from ..ordering import order_key_for_subtree
from ..transition_state import PivotTransitionState

logger = logging.getLogger(__name__)


def _assign_unclaimed_expands_to_final_mover(
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    all_expand_splits: PartitionSet[Partition],
    current_pivot_edge: Partition,
    subtree_order_key: Optional[Mapping[Partition, Tuple[int, ...]]],
) -> None:
    claimed_expands: PartitionSet[Partition] = PartitionSet(
        (
            set().union(*expand_splits_by_subtree.values())
            if expand_splits_by_subtree
            else set()
        ),
        encoding=all_expand_splits.encoding,
    )
    unassigned_expands = all_expand_splits - claimed_expands

    if not unassigned_expands:
        return

    target_subtree = (
        max(
            expand_splits_by_subtree.keys(),
            key=lambda partition: order_key_for_subtree(partition, subtree_order_key),
        )
        if expand_splits_by_subtree
        else current_pivot_edge
    )
    if target_subtree not in expand_splits_by_subtree:
        expand_splits_by_subtree[target_subtree] = PartitionSet(
            encoding=all_expand_splits.encoding
        )
    expand_splits_by_subtree[target_subtree] = (
        expand_splits_by_subtree[target_subtree] | unassigned_expands
    )
    logger.debug(
        "[planner] pivot=%s assigning %d unclaimed expands to final mover=%s",
        current_pivot_edge.bipartition(),
        len(unassigned_expands),
        target_subtree.bipartition(),
    )


def build_pivot_transition_plan(
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_tree: Node,
    expand_tree: Node,
    current_pivot_edge: Partition,
    subtree_order_key: Optional[Mapping[Partition, Tuple[int, ...]]] = None,
) -> PivotTransitionPlan:
    """Assign collapse/expand split work to ordered mover subtrees."""
    plans: PivotTransitionPlan = OrderedDict()

    all_collapse_splits, all_expand_splits = (
        get_unique_splits_for_current_pivot_edge_subtree(
            collapse_tree,
            expand_tree,
            current_pivot_edge,
        )
    )

    _assign_unclaimed_expands_to_final_mover(
        expand_splits_by_subtree,
        all_expand_splits,
        current_pivot_edge,
        subtree_order_key,
    )

    state = PivotTransitionState(
        all_collapse_splits,
        all_expand_splits,
        collapse_splits_by_subtree,
        expand_splits_by_subtree,
        current_pivot_edge,
        subtree_order_key=subtree_order_key,
    )

    logger.debug(
        "[planner] pivot=%s all_expand_splits=%s expand_paths_by_subtree=%s",
        current_pivot_edge.bipartition(),
        [list(partition.indices) for partition in all_expand_splits],
        {
            subtree.bipartition(): [list(partition.indices) for partition in splits]
            for subtree, splits in expand_splits_by_subtree.items()
        },
    )

    while state.has_remaining_work():
        subtree: Partition | None = state.get_next_subtree()
        if subtree is None:
            break

        split_classes = gather_subtree_transition_splits(state, subtree)
        is_first_subtree = not state.first_subtree_processed

        prospective_expand = (
            split_classes.last_user_expand | split_classes.unique_expand
        )
        incompatible = find_incompatible_splits(
            prospective_expand, state.all_collapsible_splits
        )

        collapse_path = assemble_collapse_path(
            split_classes.shared_collapse,
            split_classes.unique_collapse,
            incompatible,
        )

        split_classes.contingent_expand |= (
            state.consume_contingent_expand_splits_for_subtree(
                subtree=subtree, collapsed_splits=collapse_path
            )
        )

        expand_path = assemble_expand_path(
            split_classes.last_user_expand,
            split_classes.unique_expand,
            split_classes.contingent_expand,
        )

        if is_first_subtree:
            state.mark_first_subtree_processed()

        collapse_path, expand_path = store_ordered_transition_step(
            plans, state, subtree, collapse_path, expand_path
        )
        mark_transition_step_processed(state, subtree, collapse_path, expand_path)

    return plans
