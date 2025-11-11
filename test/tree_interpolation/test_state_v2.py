"""
Test suite for state_v2.py - Pre-categorized implementation.

This test suite verifies that the new pre-categorized implementation
maintains the same behavior as the original dynamic implementation.
"""

import unittest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.state_v2 import (
    InterpolationState,
    categorize_splits,
)


class TestCategorizeSplits(unittest.TestCase):
    """Test the categorize_splits helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.encoding = {"A": 0, "B": 1, "C": 2}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)

    def test_categorize_all_unique_splits(self):
        """All splits unique when each subtree has different splits."""
        splits_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        unique, shared = categorize_splits(splits_by_subtree)

        self.assertEqual(len(unique), 2)
        self.assertEqual(len(shared), 0)
        self.assertEqual(unique[self.part_A], self.part_A)
        self.assertEqual(unique[self.part_B], self.part_B)

    def test_categorize_all_shared_splits(self):
        """All splits shared when subtrees have same splits."""
        splits_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        unique, shared = categorize_splits(splits_by_subtree)

        self.assertEqual(len(unique), 0)
        self.assertEqual(len(shared), 1)
        self.assertIn(self.part_AB, shared)
        self.assertEqual(shared[self.part_AB], {self.part_A, self.part_B})

    def test_categorize_mixed_splits(self):
        """Mix of unique and shared splits."""
        splits_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        unique, shared = categorize_splits(splits_by_subtree)

        # part_AB is shared, others are unique
        self.assertEqual(len(unique), 3)
        self.assertEqual(len(shared), 1)
        self.assertIn(self.part_AB, shared)
        self.assertEqual(shared[self.part_AB], {self.part_A, self.part_B})
        self.assertEqual(unique[self.part_A], self.part_A)
        self.assertEqual(unique[self.part_B], self.part_B)
        self.assertEqual(unique[self.part_C], self.part_C)


class TestInterpolationStateV2(unittest.TestCase):
    """Test the new pre-categorized InterpolationState implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_initialization_categorizes_splits(self):
        """State correctly categorizes splits during initialization."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        state = InterpolationState(
            PartitionSet([self.part_AB, self.part_A], encoding=self.encoding),
            PartitionSet([self.part_A], encoding=self.encoding),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        # part_AB is shared collapse (2 users)
        self.assertIn(self.part_AB, state.shared_collapse_splits)
        self.assertEqual(
            state.shared_collapse_splits[self.part_AB], {self.part_A, self.part_B}
        )

        # part_A collapse is unique to part_A
        self.assertIn(self.part_A, state.unique_collapse_splits)
        self.assertEqual(state.unique_collapse_splits[self.part_A], self.part_A)

        # part_A expand is unique
        self.assertIn(self.part_A, state.unique_expand_splits)

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

        state = InterpolationState(
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
        """Last user detection works with pre-categorized approach."""
        expand_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
        }

        state = InterpolationState(
            PartitionSet(encoding=self.encoding),
            PartitionSet(
                [self.part_AB, self.part_A, self.part_B], encoding=self.encoding
            ),
            {},
            expand_by_subtree,
            self.part_ABCD,
        )

        # Initially, neither is last user for SHARED splits (both have part_AB)
        last_user_A = state.get_expand_splits_for_last_user(self.part_A)
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)

        # get_expand_splits_for_last_user only returns SHARED splits, not unique ones
        # part_A and part_B are unique, so they won't be in last_user results
        self.assertEqual(
            len(last_user_A), 0, "No shared splits where A is last user yet"
        )
        self.assertEqual(
            len(last_user_B), 0, "No shared splits where B is last user yet"
        )

        # Verify unique splits are separate
        unique_A = state.get_unique_expand_splits(self.part_A)
        unique_B = state.get_unique_expand_splits(self.part_B)
        self.assertIn(self.part_A, unique_A)
        self.assertIn(self.part_B, unique_B)

        # Process part_A's expand splits (including its share of part_AB)
        state._process_expand_split(self.part_AB, self.part_A)
        state._process_expand_split(self.part_A, self.part_A)

        # Now part_B should be last user of part_AB
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)
        self.assertIn(
            self.part_AB, last_user_B, "part_B is now last user of shared split part_AB"
        )

    def test_split_processing_updates_categories(self):
        """Processing splits correctly updates categorization."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        state = InterpolationState(
            PartitionSet([self.part_AB], encoding=self.encoding),
            PartitionSet(encoding=self.encoding),
            collapse_by_subtree,
            {},
            self.part_ABCD,
        )

        # part_AB is shared
        self.assertIn(self.part_AB, state.shared_collapse_splits)

        # Delete it
        state._delete_collapse_split(self.part_AB)

        # Now it's gone
        self.assertNotIn(self.part_AB, state.shared_collapse_splits)
        self.assertNotIn(self.part_AB, state.unique_collapse_splits)

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

        state = InterpolationState(
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
