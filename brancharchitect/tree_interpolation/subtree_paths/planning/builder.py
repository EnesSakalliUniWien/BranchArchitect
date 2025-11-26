from typing import Dict, Any
from collections import OrderedDict
import logging
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .pivot_split_registry import PivotSplitRegistry

from ..analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
    find_incompatible_splits,
)
from brancharchitect.tree import Node

logger = logging.getLogger(__name__)


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
    state: PivotSplitRegistry, subtree: Partition
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

    return {
        "shared_collapse": shared_collapse,
        "unique_collapse": unique_collapse,
        "last_user_expand": last_user_expand,
        "unique_expand": unique_expand,
        "contingent_expand": PartitionSet(encoding=state.encoding),
    }


def _finalize_and_store_plan(
    plans: OrderedDict[Partition, Dict[str, Any]],
    state: PivotSplitRegistry,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> None:
    """Handles last subtree logic, sorts paths, and stores the plan."""
    if state.is_last_subtree(subtree):
        expand_path |= state.get_all_remaining_expand_splits()
        collapse_path |= state.get_all_remaining_collapse_splits()

    # Deterministic path ordering (larger partitions first)
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
    state: PivotSplitRegistry,
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
    # Include contingent splits in processed_expand since they're now tracked in expand_tracker
    processed_expand = (
        splits["last_user_expand"]
        | splits["unique_expand"]
        | splits["contingent_expand"]
    )

    # Mark splits as processed in the state - this will remove shared splits from all subtrees
    # Note: This also marks the subtree as processed to prevent reprocessing
    state.mark_splits_as_processed(
        subtree=subtree,
        processed_collapse_splits=processed_collapse,
        processed_expand_splits=processed_expand,
    )


def build_edge_plan(
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_tree: Node,
    expand_tree: Node,
    current_pivot_edge: Partition,
) -> OrderedDict[Partition, Dict[str, Any]]:
    """Build execution plan for a pivot edge by assigning splits to subtrees.

    Note: expand_splits_by_subtree contains PATH-based assignments (splits on the path
    between subtree and pivot). This may not cover ALL splits in the pivot edge subtree,
    so we ensure completeness below.
    """
    plans: OrderedDict[Partition, Dict[str, Any]] = OrderedDict()

    # Get splits within the active changing edge scope only
    all_collapse_splits, all_expand_splits = (
        get_unique_splits_for_current_pivot_edge_subtree(
            collapse_tree,
            expand_tree,
            current_pivot_edge,
        )
    )

    # COMPLETENESS GUARANTEE: Path-based assignments may miss splits not on any path
    # (e.g., contingent splits from jumping taxa, cross-branch splits). Assign any
    # unassigned expands to first subtree as fallback to ensure every split is processed.
    claimed_expands = PartitionSet(
        set().union(*expand_splits_by_subtree.values())
        if expand_splits_by_subtree
        else set(),
        encoding=all_expand_splits.encoding,
    )
    unassigned_expands = all_expand_splits - claimed_expands
    if unassigned_expands:
        first_subtree = (
            next(iter(expand_splits_by_subtree.keys()))
            if expand_splits_by_subtree
            else current_pivot_edge
        )
        if first_subtree not in expand_splits_by_subtree:
            expand_splits_by_subtree[first_subtree] = PartitionSet(
                encoding=all_expand_splits.encoding
            )
        # Reassign with a new PartitionSet to avoid in-place quirks
        expand_splits_by_subtree[first_subtree] = (
            expand_splits_by_subtree[first_subtree] | unassigned_expands
        )
        logger.debug(
            "[builder] pivot=%s assigning %d unclaimed expands to subtree=%s",
            current_pivot_edge.bipartition(),
            len(unassigned_expands),
            first_subtree.bipartition(),
        )

    # Initialize state management for proper shared splits handling
    state = PivotSplitRegistry(
        all_collapse_splits,
        all_expand_splits,
        collapse_splits_by_subtree,
        expand_splits_by_subtree,
        current_pivot_edge,
    )
    logger.debug(
        "[builder] pivot=%s all_expand_splits=%s expand_paths_by_subtree=%s",
        current_pivot_edge.bipartition(),
        [list(p.indices) for p in all_expand_splits],
        {
            st.bipartition(): [list(p.indices) for p in splits]
            for st, splits in expand_splits_by_subtree.items()
        },
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
            all_collapse_splits = state.get_tabula_rasa_collapse_splits()
            if len(all_collapse_splits) > 0:
                collapse_path = all_collapse_splits
            else:
                # Compute incompatibilities for this subtree's planned expands
                # NOTE: Don't include contingent_expand here - they haven't been consumed yet!
                prospective_expand = (
                    splits["last_user_expand"] | splits["unique_expand"]
                )
                incompatible = find_incompatible_splits(
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
            # NOTE: Don't include contingent_expand here - they haven't been consumed yet!
            prospective_expand = splits["last_user_expand"] | splits["unique_expand"]
            incompatible = find_incompatible_splits(
                prospective_expand, state.all_collapsible_splits
            )

            collapse_path = build_collapse_path(
                splits["shared_collapse"],
                splits["unique_collapse"],
                incompatible,
            )

        # After determining the actual collapse path (including tabula rasa or
        # incompatibility collapses), consume contingent splits that fit within
        # ANY collapsed region.
        extra_contingent = state.consume_contingent_expand_splits_for_subtree(
            subtree=subtree, collapsed_splits=collapse_path
        )
        splits["contingent_expand"] |= extra_contingent

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
