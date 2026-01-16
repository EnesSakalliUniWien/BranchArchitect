import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree import Node


class MockNode:
    def __init__(self, splits):
        self._splits = splits

    def find_node_by_split(self, split):
        # Return a mock node-like object that returns the splits
        class MockSubNode:
            def to_splits(self_inner):
                return self._splits

        return MockSubNode()


def test_get_unique_splits_handles_complements():
    # 5 Taxa: A, B, C, D, E. Indices: 0, 1, 2, 3, 4.
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Source has split {A, B} -> Indices {0, 1}
    # Destination has split {C, D, E} -> Indices {2, 3, 4}
    # These are COMPLEMENTS in a 5-taxon universe. They represent the SAME unrooted split.

    src_p = Partition((0, 1), encoding)
    dst_p = Partition((2, 3, 4), encoding)

    # Pivot Edge (dummy)
    pivot = Partition((0,), encoding)

    src_splits = PartitionSet({src_p}, encoding=encoding)
    dst_splits = PartitionSet({dst_p}, encoding=encoding)

    # Mock Trees
    src_tree = MockNode(src_splits)
    dst_tree = MockNode(dst_splits)

    # Test Function
    unique_src, unique_dst = get_unique_splits_for_current_pivot_edge_subtree(
        src_tree, dst_tree, pivot
    )

    # Expectation: logic should identify them as equivalent and remove them from "Unique"
    assert len(unique_src) == 0, f"Expected 0 unique source splits, got {unique_src}"
    assert len(unique_dst) == 0, f"Expected 0 unique dest splits, got {unique_dst}"


def test_get_unique_splits_keeps_truly_unique():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Source: {A, B}
    # Dest: {A, C} (Not complement, disjoint but distinct)

    src_p = Partition((0, 1), encoding)
    dst_p = Partition((0, 2), encoding)  # {A, C} -> {0, 2}

    pivot = Partition((0,), encoding)

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
