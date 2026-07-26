import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)


class MockNode:
    def __init__(self, splits):
        self._splits = splits

    def find_node_by_split(self, split):
        # Return a mock node-like object that returns the splits
        class MockSubNode:
            def to_splits(self_inner):
                return self._splits

        return MockSubNode()


def test_get_unique_splits_keeps_pivot_complements_as_rooted_changes():
    # 5 Taxa: A, B, C, D, E. Indices: 0, 1, 2, 3, 4.
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Source has split {A, B} -> Indices {0, 1}
    # Destination has split {C, D, E} -> Indices {2, 3, 4}
    # These are complements in a 5-taxon universe, but the interpolation topology
    # planner works with rooted clades. They must remain explicit changes.

    src_p = Partition((0, 1), encoding)
    dst_p = Partition((2, 3, 4), encoding)

    pivot = Partition((0, 1, 2, 3, 4), encoding)

    src_splits = PartitionSet({src_p}, encoding=encoding)
    dst_splits = PartitionSet({dst_p}, encoding=encoding)

    # Mock Trees
    src_tree = MockNode(src_splits)
    dst_tree = MockNode(dst_splits)

    # Test Function
    unique_src, unique_dst = get_unique_splits_for_current_pivot_edge_subtree(
        src_tree, dst_tree, pivot
    )

    assert src_p in unique_src
    assert dst_p in unique_dst


def test_get_unique_splits_keeps_balanced_pivot_complements_as_rooted_changes():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    src_p = Partition((0, 1), encoding)
    dst_p = Partition((2, 3), encoding)
    pivot = Partition((0, 1, 2, 3), encoding)

    src_tree = MockNode(PartitionSet({src_p}, encoding=encoding))
    dst_tree = MockNode(PartitionSet({dst_p}, encoding=encoding))

    unique_src, unique_dst = get_unique_splits_for_current_pivot_edge_subtree(
        src_tree, dst_tree, pivot
    )

    assert src_p in unique_src
    assert dst_p in unique_dst


def test_get_unique_splits_keeps_unbalanced_pivot_complements():
    encoding = {name: index for index, name in enumerate("ABCDEFGH")}

    src_p = Partition((0, 1, 2, 3, 4, 5), encoding)
    dst_p = Partition((6, 7), encoding)
    pivot = Partition(tuple(range(8)), encoding)

    src_tree = MockNode(PartitionSet({src_p}, encoding=encoding))
    dst_tree = MockNode(PartitionSet({dst_p}, encoding=encoding))

    unique_src, unique_dst = get_unique_splits_for_current_pivot_edge_subtree(
        src_tree, dst_tree, pivot
    )

    assert src_p in unique_src
    assert dst_p in unique_dst


def test_get_unique_splits_keeps_truly_unique():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Source: {A, B}
    # Dest: {A, C} (Not complement, disjoint but distinct)

    src_p = Partition((0, 1), encoding)
    dst_p = Partition((0, 2), encoding)  # {A, C} -> {0, 2}

    pivot = Partition((0, 1, 2, 3, 4), encoding)

    src_splits = PartitionSet({src_p}, encoding=encoding)
    dst_splits = PartitionSet({dst_p}, encoding=encoding)

    src_tree = MockNode(src_splits)
    dst_tree = MockNode(dst_splits)

    unique_src, unique_dst = get_unique_splits_for_current_pivot_edge_subtree(
        src_tree, dst_tree, pivot
    )

    assert len(unique_src) == 1
    assert src_p in unique_src
    assert len(unique_dst) == 1
    assert dst_p in unique_dst


if __name__ == "__main__":
    pytest.main([__file__])
