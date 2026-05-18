"""Initialize split claims for one pivot transition."""

from __future__ import annotations

from typing import AbstractSet, Mapping

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

from .split_claim_tracker import SplitClaimTracker


def claim_valid_transition_splits(
    tracker: SplitClaimTracker,
    splits_by_subtree: Mapping[Partition, AbstractSet[Partition]],
    allowed_splits: PartitionSet[Partition],
) -> None:
    """Claim only globally valid transition splits from local subtree paths."""
    for subtree, splits in splits_by_subtree.items():
        valid_splits: PartitionSet[Partition] = PartitionSet(
            {split for split in splits if split in allowed_splits},
            encoding=tracker.encoding,
        )
        tracker.claim_batch(valid_splits, subtree)


def claim_containing_expand_splits(
    expand_tracker: SplitClaimTracker,
    initial_assignments: Mapping[Partition, AbstractSet[Partition]],
    all_expand_splits: PartitionSet[Partition],
) -> None:
    """
    Claim destination parent splits for every mover contained by the parent.

    Disjoint sibling splits remain unclaimed until collapse work creates
    compatible space for them.
    """
    for split in all_expand_splits:
        split_taxa = split.taxa
        for subtree in initial_assignments:
            if subtree.taxa.issubset(split_taxa):
                expand_tracker.claim(split, subtree)
