"""Debug script to understand why comprehensive test fails but our test passes."""

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)

# Setup from comprehensive test
encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
part_A = Partition((0,), encoding)
part_B = Partition((1,), encoding)
part_AB = Partition((0, 1), encoding)
part_ABCD = Partition((0, 1, 2, 3), encoding)

print("=== SETUP ===")
print(f"part_A: {part_A}")
print(f"part_B: {part_B}")
print(f"part_AB: {part_AB}")
print()

# Create expand_by_subtree
expand_by_subtree = {
    part_A: PartitionSet([part_AB, part_A], encoding=encoding),
    part_B: PartitionSet([part_AB, part_B], encoding=encoding),
}

collapse_by_subtree = {
    part_A: PartitionSet([part_A], encoding=encoding),
    part_B: PartitionSet([part_B], encoding=encoding),
}

print("=== EXPAND BY SUBTREE ===")
for subtree, splits in expand_by_subtree.items():
    print(f"  {subtree}: {[str(s) for s in splits]}")
print()

print("=== COLLAPSE BY SUBTREE ===")
for subtree, splits in collapse_by_subtree.items():
    print(f"  {subtree}: {[str(s) for s in splits]}")
print()

# Create state
state = PivotSplitRegistry(
    PartitionSet([part_A, part_B], encoding=encoding),
    PartitionSet([part_AB, part_A, part_B], encoding=encoding),
    collapse_by_subtree,
    expand_by_subtree,
    part_ABCD,
)

print("=== STATE AFTER INITIALIZATION ===")
print(
    f"Unique expand splits: {dict((str(k), str(v)) for k, v in state.expand_tracker.get_all_unique_resources().items())}"
)
print(
    f"Shared expand splits: {dict((str(k), {str(u) for u in v}) for k, v in state.expand_tracker.get_all_shared_resources().items())}"
)
print()

# Check if part_A is in collapse
print("=== COLLAPSE SPLITS ===")
print(
    f"Unique collapse splits: {dict((str(k), str(v)) for k, v in state.collapse_tracker.get_all_unique_resources().items())}"
)
print(
    f"Shared collapse splits: {dict((str(k), {str(u) for u in v}) for k, v in state.collapse_tracker.get_all_shared_resources().items())}"
)
print()

# Check incompatible splits
print("=== INCOMPATIBLE SPLITS ===")
# Note: PivotSplitRegistry no longer has all_incompatible_splits attribute
# Incompatible splits are now computed per-subtree via find_all_incompatible_splits_for_expand()
print("Incompatible splits are now computed per-subtree, not stored globally")
print()

# Get last user for part_A
print("=== GET LAST USER FOR part_A ===")
last_user_A = state.get_expand_splits_for_last_user(part_A)
print(f"Last user splits for part_A: {[str(s) for s in last_user_A]}")
print(f"Expected: [] (empty)")
print(f"Length: {len(last_user_A)}")
print()

# Get last user for part_B
print("=== GET LAST USER FOR part_B ===")
last_user_B = state.get_expand_splits_for_last_user(part_B)
print(f"Last user splits for part_B: {[str(s) for s in last_user_B]}")
print(f"Expected: [] (empty)")
print(f"Length: {len(last_user_B)}")
print()

# Check what's in last_user_A if it's not empty
if last_user_A:
    print("=== DEBUGGING: What's in last_user_A? ===")
    for split in last_user_A:
        print(f"  Split: {split}")
        print(f"  Split indices: {split.indices}")
        print(f"  Is it part_A? {split == part_A}")
        print(f"  Is it part_AB? {split == part_AB}")
        print(
            f"  Is it in unique_expand? {split in state.expand_tracker.get_all_unique_resources()}"
        )
        shared_expand_resources = state.expand_tracker.get_all_shared_resources()
        print(f"  Is it in shared_expand? {split in shared_expand_resources}")
        if split in shared_expand_resources:
            print(f"    Users: {[str(u) for u in shared_expand_resources[split]]}")
            print(f"    Number of users: {len(shared_expand_resources[split])}")
