"""
Tests for InterpolationState class.

Tests cover:
- Initialization and state setup
- Shared/unique split queries
- Split processing and deletion
- Subtree selection and prioritization
- Compatibility/incompatibility logic
- Remaining work tracking
- Contingent split management
"""

import unittest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.state import (
    InterpolationState,
)


class TestInterpolationStateInitialization(unittest.TestCase):
    """Test state initialization and basic setup."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create simple encoding for 4 taxa
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Create partitions
        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_CD = Partition((2, 3), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_initialization_with_basic_splits(self):
        """Test that state initializes correctly with basic splits."""
        all_collapse = PartitionSet([self.part_A, self.part_B], encoding=self.encoding)
        all_expand = PartitionSet([self.part_C, self.part_AB], encoding=self.encoding)

        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_C], encoding=self.encoding),
        }

        state = InterpolationState(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        self.assertEqual(state.encoding, self.encoding)
        self.assertEqual(state.all_collapsible_splits, all_collapse)
        self.assertEqual(state.all_expand_splits, all_expand)
        self.assertEqual(len(state.processed_subtrees), 0)

    def test_contingent_splits_computed_correctly(self):
        """Test that contingent splits are correctly identified."""
        all_collapse = PartitionSet([self.part_A], encoding=self.encoding)
        all_expand = PartitionSet(
            [self.part_B, self.part_C, self.part_AB], encoding=self.encoding
        )

        # Only assign part_B to a subtree
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_B], encoding=self.encoding),
        }

        state = InterpolationState(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        # part_C and part_AB should be contingent (not assigned to any subtree)
        self.assertIn(self.part_C, state.available_contingent_splits)
        self.assertIn(self.part_AB, state.available_contingent_splits)
        self.assertNotIn(self.part_B, state.available_contingent_splits)


class TestSharedAndUniqueSplits(unittest.TestCase):
    """Test shared/unique split queries."""

    def setUp(self):
        """Set up fixtures with shared splits."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

        # Shared split: part_A appears in both subtrees
        self.collapse_by_subtree = {
            self.part_B: PartitionSet(
                [self.part_A, self.part_B], encoding=self.encoding
            ),
            self.part_C: PartitionSet(
                [self.part_A, self.part_C], encoding=self.encoding
            ),
        }

        self.expand_by_subtree = {
            self.part_B: PartitionSet([self.part_AB], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        self.state = InterpolationState(
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            PartitionSet([self.part_AB], encoding=self.encoding),
            self.collapse_by_subtree,
            self.expand_by_subtree,
            self.part_ABCD,
        )

    def test_get_shared_collapse_splits(self):
        """Test identification of shared collapse splits."""
        shared_for_B = self.state.get_available_shared_collapse_splits(self.part_B)
        shared_for_C = self.state.get_available_shared_collapse_splits(self.part_C)

        # part_A is shared between both subtrees
        self.assertIn(self.part_A, shared_for_B)
        self.assertIn(self.part_A, shared_for_C)

        # part_B and part_C are unique
        self.assertNotIn(self.part_B, shared_for_B)
        self.assertNotIn(self.part_C, shared_for_C)

    def test_get_unique_collapse_splits(self):
        """Test identification of unique collapse splits."""
        unique_for_B = self.state.get_unique_collapse_splits(self.part_B)
        unique_for_C = self.state.get_unique_collapse_splits(self.part_C)

        # Each subtree has one unique split
        self.assertIn(self.part_B, unique_for_B)
        self.assertNotIn(self.part_A, unique_for_B)

        self.assertIn(self.part_C, unique_for_C)
        self.assertNotIn(self.part_A, unique_for_C)

    def test_get_shared_expand_splits(self):
        """Test identification of shared expand splits."""
        shared_expand_B = self.state.get_available_shared_expand_splits(self.part_B)
        shared_expand_C = self.state.get_available_shared_expand_splits(self.part_C)

        # part_AB is shared
        self.assertIn(self.part_AB, shared_expand_B)
        self.assertIn(self.part_AB, shared_expand_C)

    def test_get_expand_splits_for_last_user(self):
        """Test that last user gets splits that are about to run out."""
        # Initially, part_AB is shared (count=2), so neither is last user
        last_user_B = self.state.get_expand_splits_for_last_user(self.part_B)
        last_user_C = self.state.get_expand_splits_for_last_user(self.part_C)

        self.assertEqual(len(last_user_B), 0)
        self.assertEqual(len(last_user_C), 0)

        # After removing part_AB from one subtree, the other becomes last user
        self.state.expand_splits_by_subtree[self.part_B].discard(self.part_AB)

        last_user_C = self.state.get_expand_splits_for_last_user(self.part_C)
        self.assertIn(self.part_AB, last_user_C)


class TestSplitProcessing(unittest.TestCase):
    """Test split deletion and processing."""

    def setUp(self):
        """Set up basic state."""
        self.encoding = {"A": 0, "B": 1, "C": 2}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)

        self.collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_A, self.part_B], encoding=self.encoding
            ),
            self.part_C: PartitionSet(
                [self.part_A, self.part_C], encoding=self.encoding
            ),
        }

        self.expand_by_subtree = {
            self.part_A: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        self.state = InterpolationState(
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            PartitionSet([self.part_B, self.part_C], encoding=self.encoding),
            self.collapse_by_subtree,
            self.expand_by_subtree,
            self.part_ABC,
        )

    def test_delete_collapse_split_removes_from_all_subtrees(self):
        """Test that deleting a collapse split removes it from all subtrees."""
        self.state._delete_collapse_split(self.part_A)

        # part_A should be removed from both subtrees
        self.assertNotIn(
            self.part_A, self.state.collapse_splits_by_subtree[self.part_A]
        )
        self.assertNotIn(
            self.part_A, self.state.collapse_splits_by_subtree[self.part_C]
        )

    def test_mark_splits_as_processed(self):
        """Test that marking splits as processed updates state correctly."""
        processed_collapse = PartitionSet([self.part_A], encoding=self.encoding)
        processed_expand = PartitionSet([self.part_B], encoding=self.encoding)
        processed_contingent = PartitionSet(encoding=self.encoding)

        self.state.mark_splits_as_processed(
            self.part_A,
            processed_collapse,
            processed_expand,
            processed_contingent,
        )

        # part_A should be removed from collapse
        self.assertNotIn(
            self.part_A, self.state.collapse_splits_by_subtree[self.part_A]
        )
        self.assertNotIn(
            self.part_A, self.state.collapse_splits_by_subtree[self.part_C]
        )

        # part_B should be removed from expand and marked as used
        self.assertIn(self.part_B, self.state.used_expand_splits)

        # Note: processed_subtrees is updated externally by the caller (builder.py), not by this method

    def test_cleanup_empty_subtree_entries(self):
        """Test that empty subtrees are removed from dictionaries."""
        # Remove all splits from one subtree
        self.state.collapse_splits_by_subtree[self.part_A].clear()
        self.state.expand_splits_by_subtree[self.part_A].clear()

        self.state._cleanup_empty_subtree_entries()

        # Empty entries should be removed
        self.assertNotIn(self.part_A, self.state.collapse_splits_by_subtree)
        self.assertNotIn(self.part_A, self.state.expand_splits_by_subtree)


class TestSubtreeSelection(unittest.TestCase):
    """Test subtree selection and prioritization logic."""

    def setUp(self):
        """Set up state with different priority scenarios."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_D = Partition((3,), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_priority_0_shared_collapse_selected_first(self):
        """Test that subtrees with shared collapse splits have highest priority."""
        # part_A has shared collapse (part_D appears in both part_A and part_B)
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_D], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_D], encoding=self.encoding),
            self.part_C: PartitionSet(
                [self.part_C], encoding=self.encoding
            ),  # Only unique
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        state = InterpolationState(
            PartitionSet([self.part_C, self.part_D], encoding=self.encoding),
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        next_subtree = state.get_next_subtree()

        # Should select part_A or part_B (both have shared collapse)
        self.assertIn(next_subtree, [self.part_A, self.part_B])

    def test_priority_1_unique_splits_selected_second(self):
        """Test that subtrees with only unique splits have medium priority."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_A], encoding=self.encoding
            ),  # Unique only
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet(
                [self.part_D, self.part_C], encoding=self.encoding
            ),  # Will have shared expand
            self.part_C: PartitionSet(
                [self.part_D], encoding=self.encoding
            ),  # Shared expand
        }

        state = InterpolationState(
            PartitionSet([self.part_A], encoding=self.encoding),
            PartitionSet(
                [self.part_A, self.part_D, self.part_C], encoding=self.encoding
            ),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        next_subtree = state.get_next_subtree()

        # Should select part_A (unique only) before shared expand subtrees
        self.assertEqual(next_subtree, self.part_A)

    def test_returns_none_when_no_work_remaining(self):
        """Test that get_next_subtree returns None when all work is done."""
        state = InterpolationState(
            PartitionSet(encoding=self.encoding),
            PartitionSet(encoding=self.encoding),
            {},
            {},
            self.part_ABCD,
        )

        self.assertIsNone(state.get_next_subtree())


class TestContingentSplits(unittest.TestCase):
    """Test contingent split management."""

    def setUp(self):
        """Set up state with contingent splits."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_consume_contingent_splits_within_collapsed_region(self):
        """Test that contingent splits are correctly consumed."""
        all_expand = PartitionSet(
            [self.part_A, self.part_B, self.part_AB], encoding=self.encoding
        )

        # Only part_B is assigned to a subtree
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_B], encoding=self.encoding),
        }

        state = InterpolationState(
            PartitionSet([self.part_ABC], encoding=self.encoding),
            all_expand,
            {self.part_A: PartitionSet([self.part_ABC], encoding=self.encoding)},
            expand_by_subtree,
            self.part_ABCD,
        )

        # part_A and part_AB are contingent
        self.assertIn(self.part_A, state.available_contingent_splits)
        self.assertIn(self.part_AB, state.available_contingent_splits)

        # Consume contingent splits when collapsing part_ABC
        collapsed = PartitionSet([self.part_ABC], encoding=self.encoding)
        contingent = state.consume_contingent_expand_splits_for_subtree(
            self.part_A, collapsed
        )

        # All contingent splits that fit within part_ABC should be consumed
        self.assertIn(self.part_A, contingent)
        self.assertIn(self.part_AB, contingent)

    def test_contingent_splits_not_reused(self):
        """Test that used contingent splits are tracked and not reused."""
        all_expand = PartitionSet([self.part_A, self.part_B], encoding=self.encoding)

        state = InterpolationState(
            PartitionSet([self.part_ABC], encoding=self.encoding),
            all_expand,
            {self.part_A: PartitionSet([self.part_ABC], encoding=self.encoding)},
            {},
            self.part_ABCD,
        )

        # Mark part_A as used
        state.used_contingent_splits.add(self.part_A)

        # Try to consume contingent splits
        collapsed = PartitionSet([self.part_ABC], encoding=self.encoding)
        contingent = state.consume_contingent_expand_splits_for_subtree(
            self.part_A, collapsed
        )

        # part_A should not be included (already used)
        self.assertNotIn(self.part_A, contingent)


class TestRemainingWork(unittest.TestCase):
    """Test queries about remaining work."""

    def setUp(self):
        """Set up basic state."""
        self.encoding = {"A": 0, "B": 1, "C": 2}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)

        self.state = InterpolationState(
            PartitionSet([self.part_A, self.part_B], encoding=self.encoding),
            PartitionSet([self.part_C], encoding=self.encoding),
            {
                self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
                self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            },
            {
                self.part_A: PartitionSet([self.part_C], encoding=self.encoding),
            },
            self.part_ABC,
        )

    def test_has_remaining_work(self):
        """Test that has_remaining_work correctly identifies work."""
        self.assertTrue(self.state.has_remaining_work())

        # Mark all subtrees as processed
        self.state.processed_subtrees.add(self.part_A)
        self.state.processed_subtrees.add(self.part_B)

        self.assertFalse(self.state.has_remaining_work())

    def test_is_last_subtree(self):
        """Test identification of last remaining subtree."""
        self.assertFalse(self.state.is_last_subtree(self.part_A))

        # Process part_B
        self.state.processed_subtrees.add(self.part_B)

        # Now part_A should be the last
        self.assertTrue(self.state.is_last_subtree(self.part_A))

    def test_get_all_remaining_expand_splits(self):
        """Test retrieval of all remaining expand splits."""
        all_remaining = self.state.get_all_remaining_expand_splits()

        # Initially, part_C is the only expand split
        self.assertIn(self.part_C, all_remaining)

        # Mark it as used
        self.state.used_expand_splits.add(self.part_C)

        all_remaining = self.state.get_all_remaining_expand_splits()
        self.assertNotIn(self.part_C, all_remaining)


if __name__ == "__main__":
    unittest.main()
