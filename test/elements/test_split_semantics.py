from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.split_semantics import (
    same_rooted_split,
    same_unrooted_split_within_pivot,
    unrooted_difference_within_pivot,
)


def test_same_rooted_split_requires_exact_bitmask():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

    assert same_rooted_split(Partition((0, 1), encoding), Partition((0, 1), encoding))
    assert not same_rooted_split(
        Partition((0, 1), encoding), Partition((2, 3), encoding)
    )


def test_same_unrooted_split_uses_pivot_scope():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    pivot = Partition((0, 1, 2, 3), encoding)

    assert same_unrooted_split_within_pivot(
        Partition((0, 1), encoding),
        Partition((2, 3), encoding),
        pivot,
        max_complement_ratio=2,
    )


def test_unrooted_difference_preserves_unbalanced_rooted_clades():
    encoding = {name: index for index, name in enumerate("ABCDEFGH")}
    pivot = Partition(tuple(range(8)), encoding)
    large = Partition((0, 1, 2, 3, 4, 5), encoding)
    small = Partition((6, 7), encoding)

    difference = unrooted_difference_within_pivot(
        PartitionSet({large}, encoding=encoding),
        PartitionSet({small}, encoding=encoding),
        pivot,
        max_complement_ratio=2,
    )

    assert large in difference
