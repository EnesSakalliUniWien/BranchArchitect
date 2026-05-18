"""Assemble ordered collapse and expand paths for one pivot transition."""

from __future__ import annotations

from dataclasses import dataclass

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

from ..transition_state import PivotTransitionState
from .transition_step import PivotTransitionPlan, PivotTransitionStep


@dataclass
class SubtreeTransitionSplitClasses:
    """Split categories used to assemble one mover transition."""

    shared_collapse: PartitionSet[Partition]
    unique_collapse: PartitionSet[Partition]
    last_user_expand: PartitionSet[Partition]
    unique_expand: PartitionSet[Partition]
    contingent_expand: PartitionSet[Partition]


def assemble_collapse_path(
    shared_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    incompatible_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Combine all split classes that must collapse for a mover."""
    return shared_splits | unique_splits | incompatible_splits


def assemble_expand_path(
    last_user_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    contingent_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Combine all split classes that must expand for a mover."""
    return last_user_splits | unique_splits | contingent_splits


def gather_subtree_transition_splits(
    state: PivotTransitionState, subtree: Partition
) -> SubtreeTransitionSplitClasses:
    """Gather the split classes needed to build one mover transition."""
    return SubtreeTransitionSplitClasses(
        shared_collapse=state.get_available_shared_collapse_splits(subtree),
        unique_collapse=state.get_unique_collapse_splits(subtree),
        last_user_expand=state.get_expand_splits_for_last_user(subtree),
        unique_expand=state.get_unique_expand_splits(subtree),
        contingent_expand=PartitionSet(encoding=state.encoding),
    )


def store_ordered_transition_step(
    plans: PivotTransitionPlan,
    state: PivotTransitionState,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> tuple[PartitionSet[Partition], PartitionSet[Partition]]:
    """Apply final cleanup semantics, sort paths, and write one plan entry."""
    if state.is_last_subtree(subtree):
        expand_path |= state.get_all_remaining_expand_splits()
        collapse_path |= state.get_all_remaining_collapse_splits()

    plans[subtree] = PivotTransitionStep(
        subtree=subtree,
        collapse_path=tuple(
            sorted(
                collapse_path,
                key=lambda partition: (len(partition.indices), partition.bitmask),
            )
        ),
        expand_path=tuple(
            sorted(
                expand_path,
                key=lambda partition: (-len(partition.indices), partition.bitmask),
            )
        ),
    )
    return collapse_path, expand_path


def mark_transition_step_processed(
    state: PivotTransitionState,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> None:
    """Persist the executed split work into the pivot transition state."""
    state.mark_splits_as_processed(
        subtree=subtree,
        processed_collapse_splits=collapse_path,
        processed_expand_splits=expand_path,
    )
