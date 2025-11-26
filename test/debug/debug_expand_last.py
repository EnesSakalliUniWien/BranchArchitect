"""Debug test to understand what's happening with expand-last strategy"""

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)

# Setup from the test
encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
part_A = Partition((0,), encoding)
part_B = Partition((1,), encoding)
part_AB = Partition((0, 1), encoding)
part_ABCD = Partition((0, 1, 2, 3), encoding)

expand_by_subtree = {
    part_A: PartitionSet([part_AB, part_A], encoding=encoding),
    part_B: PartitionSet([part_AB, part_B], encoding=encoding),
}

collapse_by_subtree = {
    part_A: PartitionSet([part_A], encoding=encoding),
    part_B: PartitionSet([part_B], encoding=encoding),
}

state = PivotSplitRegistry(
    PartitionSet([part_A, part_B], encoding=encoding),
    PartitionSet([part_AB, part_A, part_B], encoding=encoding),
    collapse_by_subtree,
    expand_by_subtree,
    part_ABCD,
)

print("=== EXPAND SPLITS CATEGORIZATION ===")
print(
    f"Unique expand splits: {dict((str(k), str(v)) for k, v in state.expand_tracker.get_all_unique_resources().items())}"
)
print(
    f"Shared expand splits: {dict((str(k), {str(u) for u in v}) for k, v in state.expand_tracker.get_all_shared_resources().items())}"
)

print("\n=== INCOMPATIBLE SPLITS ===")
print("Incompatible splits are now computed per-subtree, not stored globally")

print("\n=== GET LAST USER FOR part_A ===")
last_user_A = state.get_expand_splits_for_last_user(part_A)
print(f"Last user splits for part_A: {[str(s) for s in last_user_A]}")
print(f"Expected: [] (empty)")

print("\n=== GET LAST USER FOR part_B ===")
last_user_B = state.get_expand_splits_for_last_user(part_B)
print(f"Last user splits for part_B: {[str(s) for s in last_user_B]}")
print(f"Expected: [] (empty)")
