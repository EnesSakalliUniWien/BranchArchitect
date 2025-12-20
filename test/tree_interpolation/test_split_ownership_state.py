"""
Test suite for pivot_split_registry.py - OwnershipTracker-based implementation.

This test suite verifies PivotSplitRegistry functionality.
Note: The analyze_split_ownership helper function has been removed in favor
of OwnershipTracker, which is tested in test_ownership_tracker.py.

TODO: Some tests in this file need updating to work with OwnershipTracker API
instead of directly accessing removed internal dicts (shared_collapse_splits, etc).
"""

import unittest
import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)


# NOTE: TestCategorizeSplits has been removed as analyze_split_ownership()
# function no longer exists. OwnershipTracker provides this functionality
# and is tested comprehensively in test_ownership_tracker.py.


class TestInterpolationStateV2(unittest.TestCase):
    """Test the new pre-categorized PivotSplitRegistry implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_get_unique_and_shared_queries(self):
        """Query methods return correct categorized splits."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet(
                [self.part_AB, self.part_A, self.part_B], encoding=self.encoding
            ),
            PartitionSet([self.part_A], encoding=self.encoding),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        # Get unique collapse for part_A
        unique_collapse_A = state.get_unique_collapse_splits(self.part_A)
        self.assertIn(self.part_A, unique_collapse_A)
        self.assertNotIn(self.part_AB, unique_collapse_A)

        # Get shared collapse for part_A
        shared_collapse_A = state.get_available_shared_collapse_splits(self.part_A)
        self.assertIn(self.part_AB, shared_collapse_A)

        # Get unique expand for part_A
        unique_expand_A = state.get_unique_expand_splits(self.part_A)
        self.assertIn(self.part_A, unique_expand_A)

    def test_expand_last_strategy(self):
        """Last user detection works with OwnershipTracker."""
        expand_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
        }

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            PartitionSet(
                [self.part_AB, self.part_A, self.part_B], encoding=self.encoding
            ),
            {},
            expand_by_subtree,
            self.part_ABCD,
        )

        # Initially, each subtree is the last (and only) user of its unique splits
        # part_A is unique to subtree A, part_B is unique to subtree B
        last_user_A = state.get_expand_splits_for_last_user(self.part_A)
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)

        # Unique splits ARE returned as "last user" (owner_count=1)
        self.assertIn(self.part_A, last_user_A, "part_A is unique, so A is last owner")
        self.assertIn(self.part_B, last_user_B, "part_B is unique, so B is last owner")

        # Shared split part_AB should NOT be in last_user results yet (both own it)
        self.assertNotIn(
            self.part_AB, last_user_A, "part_AB is shared, A is not last user"
        )
        self.assertNotIn(
            self.part_AB, last_user_B, "part_AB is shared, B is not last user"
        )

        # Process part_A's expand splits (release ownership)
        state.expand_tracker.release(self.part_AB, self.part_A)
        state.expand_tracker.release(self.part_A, self.part_A)

        # Now part_B should be last user of part_AB
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)
        self.assertIn(
            self.part_AB, last_user_B, "part_B is now last user of shared split part_AB"
        )

    def test_subtree_selection_based_on_expand_length(self):
        """Subtree selection prioritizes longest expand paths."""
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),  # 1 split
            self.part_B: PartitionSet(
                [self.part_B, self.part_AB], encoding=self.encoding
            ),  # 2 splits
            self.part_C: PartitionSet(
                [self.part_C, self.part_ABC, self.part_AB], encoding=self.encoding
            ),  # 3 splits
        }

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            PartitionSet(
                [self.part_A, self.part_B, self.part_C, self.part_AB, self.part_ABC],
                encoding=self.encoding,
            ),
            {},
            expand_by_subtree,
            self.part_ABCD,
        )

        # Should select part_C (3 expand splits)
        next_subtree = state.get_next_subtree()
        self.assertEqual(next_subtree, self.part_C)


if __name__ == "__main__":
    unittest.main()
