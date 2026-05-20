"""Explicit rooted/unrooted split comparison helpers."""

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


def same_rooted_split(left: Partition, right: Partition) -> bool:
    """Return True only when two rooted splits are the exact same side."""
    return left.bitmask == right.bitmask


def same_unrooted_split_within_pivot(
    left: Partition,
    right: Partition,
    pivot: Partition,
    *,
    max_complement_ratio: int | None = None,
) -> bool:
    """
    Compare splits as unrooted complements within a pivot/subtree universe.

    `max_complement_ratio` keeps highly unbalanced complement pairs distinct
    when callers still need rooted subtree ownership.
    """
    if same_rooted_split(left, right):
        return True

    left_mask = left.bitmask
    right_mask = right.bitmask
    pivot_mask = pivot.bitmask
    if (left_mask | right_mask) != pivot_mask or left_mask & right_mask:
        return False

    if max_complement_ratio is None:
        return True

    smaller_size = min(left.size, right.size)
    larger_size = max(left.size, right.size)
    return larger_size <= smaller_size * max_complement_ratio


def unrooted_difference_within_pivot(
    left: PartitionSet[Partition],
    right: PartitionSet[Partition],
    pivot: Partition,
    *,
    max_complement_ratio: int | None = None,
) -> PartitionSet[Partition]:
    """Return `left - right` using pivot-scoped unrooted split equivalence."""
    return PartitionSet(
        {
            split
            for split in left
            if not any(
                same_unrooted_split_within_pivot(
                    split,
                    right_split,
                    pivot,
                    max_complement_ratio=max_complement_ratio,
                )
                for right_split in right
            )
        },
        encoding=left.encoding,
    )
