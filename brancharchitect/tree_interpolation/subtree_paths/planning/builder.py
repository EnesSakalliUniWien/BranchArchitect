from typing import Dict, Any
from collections import OrderedDict
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .state_v2 import InterpolationState

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

    # Deterministic path ordering (larger partitions first)
    collapse_path_list = sorted(collapse_path, key=lambda p: len(p.indices), reverse=True)
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
    collapse_path: PartitionSet[Partition],
) -> None:
    """Marks splits and the subtree as processed in the state.

    Args:
        state: The interpolation state to update
        subtree: The subtree being processed
        splits: Dictionary of split categories for this subtree
        incompatible_splits: Incompatible splits that were identified
        collapse_path: The ACTUAL collapse path that will be executed (may be ALL splits for TABULA RASA)
    """
    # Use the actual collapse_path that will be executed (covers TABULA RASA first subtree)
    processed_collapse = collapse_path
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
    current_pivot_edge: Partition,
) -> OrderedDict[Partition, Dict[str, Any]]:
    plans: OrderedDict[Partition, Dict[str, Any]] = OrderedDict()

    # Get splits within the active changing edge scope only
    all_collapse_splits, all_expand_splits = (
        get_unique_splits_for_active_changing_edge_subtree(
            collapse_tree,
            expand_tree,
            current_pivot_edge,
        )
    )

    # Initialize state management for proper shared splits handling
    state = InterpolationState(
        all_collapse_splits,
        all_expand_splits,
        collapse_splits_by_subtree,
        expand_splits_by_subtree,
        current_pivot_edge,
    )

    while state.has_remaining_work():
        # Get next subtree using priority algorithm
        subtree: Partition | None = state.get_next_subtree()

        if subtree is None:
            break

        # 1. Gather all component splits for the current subtree
        splits = _gather_subtree_splits(state, subtree)

        # ========================================================================
        # 2. TABULA RASA STRATEGY
        # ========================================================================
        # First subtree may collapse everything (tabula rasa) only if there are
        # actual collapses at this pivot edge. Otherwise, respect per-subtree
        # assignments even for the first subtree.
        is_first_subtree = not state.first_subtree_processed

        collapse_path: PartitionSet[Partition]
        incompatible: PartitionSet[Partition] = PartitionSet(encoding=state.encoding)
        if is_first_subtree:
            all_collapse_splits = state.get_all_collapse_splits_for_first_subtree(
                subtree
            )
            if len(all_collapse_splits) > 0:
                collapse_path = all_collapse_splits
            else:
                # Compute incompatibilities for this subtree's planned expands
                prospective_expand = (
                    splits["last_user_expand"]
                    | splits["unique_expand"]
                    | splits["contingent_expand"]
                )
                incompatible = state.find_all_incompatible_splits_for_expand(
                    prospective_expand, state.all_collapsible_splits
                )

                collapse_path = build_collapse_path(
                    splits["shared_collapse"],
                    splits["unique_collapse"],
                    incompatible,
                )
        else:
            # Subsequent subtrees: only their assigned splits
            # Compute incompatibilities for this subtree's planned expands
            prospective_expand = (
                splits["last_user_expand"]
                | splits["unique_expand"]
                | splits["contingent_expand"]
            )
            incompatible = state.find_all_incompatible_splits_for_expand(
                prospective_expand, state.all_collapsible_splits
            )

            collapse_path = build_collapse_path(
                splits["shared_collapse"],
                splits["unique_collapse"],
                incompatible,
            )

        # Build expand path (all subtrees get their expand work)
        expand_path: PartitionSet[Partition] = build_expand_path(
            splits["last_user_expand"],
            splits["unique_expand"],
            splits["contingent_expand"],
        )

        # Mark first subtree as processed
        if is_first_subtree:
            state.mark_first_subtree_processed()

        # Finalize the plan for this subtree
        _finalize_and_store_plan(plans, state, subtree, collapse_path, expand_path)

        # Update the global state - pass the actual collapse_path for TABULA RASA handling
        _update_state(
            state,
            subtree,
            splits,
            incompatible,
            collapse_path,
        )

    return plans
