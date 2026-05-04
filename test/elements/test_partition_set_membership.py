import pytest

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


def test_partition_membership_rejects_matching_bitmask_with_different_encoding():
    partition_set = PartitionSet({Partition((0,), {"A": 0})})

    with pytest.raises(ValueError, match="different encoding"):
        Partition((0,), {"B": 0}) in partition_set


def test_partition_membership_accepts_matching_bitmask_with_same_encoding():
    encoding = {"A": 0}
    partition_set = PartitionSet({Partition((0,), encoding)})

    assert Partition((0,), {"A": 0}) in partition_set


def test_partition_membership_miss_skips_encoding_validation():
    partition_set = PartitionSet({Partition((0,), {"A": 0, "B": 1})})

    assert Partition((1,), {"X": 0, "Y": 1}) not in partition_set
