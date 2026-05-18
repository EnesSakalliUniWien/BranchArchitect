"""Claim expand splits that become available after collapse work."""

from __future__ import annotations

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

from .split_claim_tracker import SplitClaimTracker


def claim_contingent_expand_splits(
    expand_tracker: SplitClaimTracker,
    all_expand_splits: PartitionSet[Partition],
    subtree: Partition,
    collapsed_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """
    Atomically claim unassigned destination splits that fit the collapsed region.
    """
    tracked_resources = expand_tracker.get_all_resources()

    if not collapsed_splits:
        unassigned_splits: PartitionSet[Partition] = PartitionSet(
            {split for split in all_expand_splits if split not in tracked_resources},
            encoding=expand_tracker.encoding,
        )
        for split in unassigned_splits:
            expand_tracker.claim(split, subtree)
        return unassigned_splits

    collapsed_regions = [set(split.indices) for split in collapsed_splits]

    contingent_splits: PartitionSet[Partition] = PartitionSet(
        encoding=expand_tracker.encoding
    )
    for expand_split in all_expand_splits:
        if expand_split in tracked_resources:
            continue
        expand_indices = set(expand_split.indices)
        if any(expand_indices.issubset(region) for region in collapsed_regions):
            contingent_splits.add(expand_split)

    for split in contingent_splits:
        expand_tracker.claim(split, subtree)

    return contingent_splits
