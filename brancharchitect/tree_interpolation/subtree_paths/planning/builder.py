from typing import Dict, Any
from collections import OrderedDict
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .state import InterpolationState

from ..analysis.split_analysis import (
    get_unique_splits_for_active_changing_edge_subtree,
)
from brancharchitect.tree import Node


# ============================================================================
# Helper Functions for Path Building
# ============================================================================


def build_collapse_path(
    shared_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    incompatible_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Build the collapse path from component splits using set operations."""
    return shared_splits | unique_splits | incompatible_splits


def build_expand_path(
    shared_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    contingent_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Build the expand path from component splits using set operations."""
    return shared_splits | unique_splits | contingent_splits


def _gather_subtree_splits(
    state: InterpolationState, subtree: Partition
) -> Dict[str, PartitionSet[Partition]]:
    """Gathers all necessary split sets for a subtree from the state."""
    shared_collapse: PartitionSet[Partition] = (
        state.get_available_shared_collapse_splits(subtree)
    )
    unique_collapse: PartitionSet[Partition] = state.get_unique_collapse_splits(subtree)
    last_user_expand: PartitionSet[Partition] = state.get_expand_splits_for_last_user(
        subtree
    )
    unique_expand: PartitionSet[Partition] = state.get_unique_expand_splits(subtree)

    # Consume contingent splits based on the collapse path
    contingent_expand = state.consume_contingent_expand_splits_for_subtree(
        subtree=subtree,
        collapsed_splits=shared_collapse | unique_collapse,
    )

    return {
        "shared_collapse": shared_collapse,
        "unique_collapse": unique_collapse,
        "last_user_expand": last_user_expand,
        "unique_expand": unique_expand,
        "contingent_expand": contingent_expand,
    }


def _finalize_and_store_plan(
    plans: OrderedDict[Partition, Dict[str, Any]],
    state: InterpolationState,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> None:
    """Handles last subtree logic, sorts paths, and stores the plan."""
    if state.is_last_subtree(subtree):
        expand_path |= state.get_all_remaining_expand_splits()
        collapse_path |= state.get_all_remaining_collapse_splits()

    # Sort by partition size for deterministic ordering (larger partitions = shallower in tree come first)
    collapse_path_list = sorted(
        collapse_path, key=lambda p: len(p.indices), reverse=True
    )
    expand_path_list = sorted(expand_path, key=lambda p: len(p.indices), reverse=True)

    # Store the full paths in the plan
    plans[subtree] = {
        "collapse": {"path_segment": collapse_path_list},
        "expand": {"path_segment": expand_path_list},
    }


def _update_state(
    state: InterpolationState,
    subtree: Partition,
    splits: Dict[str, PartitionSet[Partition]],
    incompatible_splits: PartitionSet[Partition],
) -> None:
    """Marks splits and the subtree as processed in the state."""
    processed_collapse = (
        splits["shared_collapse"] | splits["unique_collapse"] | incompatible_splits
    )
    processed_expand = splits["last_user_expand"] | splits["unique_expand"]

    # Mark splits as processed in the state - this will remove shared splits from all subtrees
    state.mark_splits_as_processed(
        subtree=subtree,
        processed_collapse_splits=processed_collapse,
        processed_expand_splits=processed_expand,
        processed_contingent_splits=splits["contingent_expand"],
    )
    # Mark the subtree as processed for this cycle to prevent infinite loops
    state.processed_subtrees.add(subtree)


def build_edge_plan(
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_tree: Node,
    expand_tree: Node,
    active_changing_edge: Partition,
) -> OrderedDict[Partition, Dict[str, Any]]:
    plans: OrderedDict[Partition, Dict[str, Any]] = OrderedDict()

    # Get splits within the active changing edge scope only
    all_collapse_splits, all_expand_splits = (
        get_unique_splits_for_active_changing_edge_subtree(
            collapse_tree,
            expand_tree,
            active_changing_edge,
        )
    )

    # Initialize state management for proper shared splits handling
    state = InterpolationState(
        all_collapse_splits,
        all_expand_splits,
        collapse_splits_by_subtree,
        expand_splits_by_subtree,
        active_changing_edge,
    )

    while state.has_remaining_work():
        # Get next subtree using priority algorithm
        subtree: Partition | None = state.get_next_subtree()

        if subtree is None:
            break

        # 1. Gather all component splits for the current subtree
        splits = _gather_subtree_splits(state, subtree)

        # ========================================================================
        # 4. Build the final path segments for this subtree
        # ========================================================================

        # Build expand path
        expand_path: PartitionSet[Partition] = build_expand_path(
            splits["last_user_expand"],
            splits["unique_expand"],
            splits["contingent_expand"],
        )

        # Find incompatible splits
        incompatible_splits: PartitionSet[Partition] = (
            state.find_all_incompatible_splits_for_expand(
                expand_partitions=(expand_path),
                all_available_collapse_splits=state.all_collapsible_splits,
            )
        )

        # Build collapse path BEFORE deleting anything
        collapse_path: PartitionSet[Partition] = build_collapse_path(
            splits["shared_collapse"],
            splits["unique_collapse"],
            incompatible_splits,  # Still available in state
        )

        # Finalize the plan for this subtree
        _finalize_and_store_plan(plans, state, subtree, collapse_path, expand_path)

        # Update the global state
        _update_state(state, subtree, splits, incompatible_splits)

    return plans
