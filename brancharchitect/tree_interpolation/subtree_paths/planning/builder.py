from typing import Dict, Any
from collections import OrderedDict
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .state import InterpolationState

from ..analysis.split_analysis import (
    get_unique_splits_for_active_changing_edge_subtree,
)
from brancharchitect.tree import Node
from .diagnostics import (
    log_final_plans,
)


# ============================================================================
# Helper Functions for Path Building
# ============================================================================


def build_collapse_path(
    shared_splits: "PartitionSet[Partition]",
    unique_splits: "PartitionSet[Partition]",
    incompatible_splits: "PartitionSet[Partition]",
) -> "PartitionSet[Partition]":
    """Build the collapse path from component splits using set operations.

    Args:
        shared_splits: Splits that are shared between subtrees
        unique_splits: Splits that are unique to this subtree
        incompatible_splits: Splits that are incompatible with expansion

    Returns:
        PartitionSet of splits forming the collapse path
    """
    # Use union to combine all splits, automatically removing duplicates
    return shared_splits | unique_splits | incompatible_splits


def build_expand_path(
    shared_splits: "PartitionSet[Partition]",
    unique_splits: "PartitionSet[Partition]",
    compatible_splits: "PartitionSet[Partition]",
) -> "PartitionSet[Partition]":
    """Build the expand path from component splits using set operations.

    Args:
        shared_splits: Splits that are shared between subtrees
        unique_splits: Splits that are unique to this subtree
        compatible_splits: Splits that are compatible with collapsed splits

    Returns:
        PartitionSet of splits forming the expand path
    """
    return shared_splits | unique_splits | compatible_splits


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

        # Get splits for this subtree using enhanced state management
        to_be_collapsed_shared_path_splits: PartitionSet[Partition] = (
            state.get_available_shared_collapse_splits(subtree)
        )

        to_be_collapsed_unique_path_splits: PartitionSet[Partition] = (
            state.get_unique_collapse_splits(subtree)
        )

        # For expand: distinguish between shared splits this subtree can use vs must use (last user)
        expand_shared_splits_last_user: PartitionSet[Partition] = (
            state.get_expand_splits_for_last_user(subtree)
        )

        expand_unique_path_splits: PartitionSet[Partition] = (
            state.get_unique_expand_splits(subtree)
        )

        # Get compatible splits: find existing expand splits that are compatible with the last collapsed split
        # Use all expand candidates to ensure orthogonal elements are considered
        compatible_expand_splits: PartitionSet[Partition] = (
            state.consume_compatible_expand_splits_for_subtree(
                subtree=subtree,
                collapsed_splits=to_be_collapsed_shared_path_splits
                | to_be_collapsed_unique_path_splits,
            )
        )

        # ========================================================================
        # 4. Build the final path segments for this subtree
        # ========================================================================

        expand_path: PartitionSet[Partition] = build_expand_path(
            expand_shared_splits_last_user,
            expand_unique_path_splits,
            compatible_expand_splits,
        )

        # Find ALL incompatible splits from the currently available splits that conflict with this subtree's expand paths
        incompatible_splits: PartitionSet[Partition] = (
            state.find_all_incompatible_splits_for_expand(
                expand_partitions=(expand_path),
                all_available_collapse_splits=state.all_collapsible_splits,
            )
        )

        # Delete incompatible splits globally from ALL subtrees (they conflict with expansion)
        if incompatible_splits:
            state.delete_global_collapse_splits(incompatible_splits)

        # Build collapse and expand paths (original behavior)
        collapse_path: PartitionSet[Partition] = build_collapse_path(
            to_be_collapsed_shared_path_splits,
            to_be_collapsed_unique_path_splits,
            incompatible_splits,
        )

        if state.is_last_subtree(subtree):
            # Final step: This is the last subtree, so it must handle ALL remaining splits to reach the target topology.
            # Collect any remaining shared, unique, and compatible expand splits.
            expand_path |= state.get_all_remaining_expand_splits()

        # Convert to sorted lists for deterministic ordering
        # Sort by partition size (larger partitions = shallower in tree come first)
        collapse_path_list = sorted(collapse_path, key=lambda p: len(p.indices), reverse=True)
        expand_path_list = sorted(expand_path, key=lambda p: len(p.indices), reverse=True)

        # Store the full paths in the plan (for visualization/execution)
        plans[subtree] = {
            "collapse": {"path_segment": collapse_path_list},
            "expand": {"path_segment": expand_path_list},
        }

        # For marking as processed, only pass splits that belong to this subtree
        # (incompatible splits were already globally deleted above)
        subtree_collapse_splits = (
            to_be_collapsed_shared_path_splits | to_be_collapsed_unique_path_splits
        )
        subtree_expand_splits = (
            expand_shared_splits_last_user | expand_unique_path_splits
        )

        # Mark splits as processed in the state - this will remove shared splits from all subtrees
        state.mark_splits_as_processed(
            subtree=subtree,
            processed_collapse_splits=subtree_collapse_splits,
            processed_expand_splits=subtree_expand_splits,
        )
        # Mark the subtree as processed for this cycle to prevent infinite loops
        state.processed_subtrees.add(subtree)

    log_final_plans(plans)

    return plans
