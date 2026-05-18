from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.ordering.path_group_manager import (
    PathGroupManager,
)


def test_visual_order_breaks_equal_ready_ties():
    encoding = {f"T{i}": i for i in range(10)}

    subtree_a = Partition((0,), encoding)
    subtree_d = Partition((3,), encoding)

    shared_split = Partition((4,), encoding)
    split_a = Partition((5,), encoding)
    split_d = Partition((6,), encoding)

    expand_splits_by_subtree = {
        subtree_a: PartitionSet({shared_split, split_a}, encoding=encoding),
        subtree_d: PartitionSet({shared_split, split_d}, encoding=encoding),
    }

    manager = PathGroupManager(
        expand_splits_by_subtree,
        encoding,
        subtree_order_key={subtree_d: (0,), subtree_a: (1,)},
    )

    assert manager.get_next_subtree(set()) == subtree_d
