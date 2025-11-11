"""
Test to verify that subtrees correctly receive their shared and unique splits.

This test verifies the expand-last strategy and split categorization in state_v2.
"""

import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.state_v2 import (
    InterpolationState,
    categorize_splits,
)


class TestSharedAndUniqueSplits:
    """Test that subtrees correctly receive their shared and unique splits."""

    @pytest.fixture
    def encoding(self):
        """Standard encoding for tests"""
        return {"A": 0, "B": 1, "C": 2, "D": 3}

    @pytest.fixture
    def partitions(self, encoding):
        """Create test partitions"""
        return {
            "part_A": Partition((0,), encoding),
            "part_B": Partition((1,), encoding),
            "part_C": Partition((2,), encoding),
            "part_AB": Partition((0, 1), encoding),
            "part_BC": Partition((1, 2), encoding),
            "part_ABC": Partition((0, 1, 2), encoding),
            "part_ABCD": Partition((0, 1, 2, 3), encoding),
        }

    def test_unique_expand_splits_returned_correctly(self, encoding, partitions):
        """
        Test that unique expand splits are correctly identified and returned.

        Setup:
            - Subtree A has unique expand splits: [A]
            - Subtree B has unique expand splits: [B]
            - Shared expand split: [AB] (used by both A and B)

        Expected:
            - get_unique_expand_splits(A) returns [A]
            - get_unique_expand_splits(B) returns [B]
            - Neither returns [AB] (that's shared)
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_AB"], partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_AB"], partitions["part_B"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_B"]}, encoding=encoding
            ),
        }

        state = InterpolationState(
            PartitionSet(
                {partitions["part_A"], partitions["part_B"]}, encoding=encoding
            ),
            PartitionSet(
                {partitions["part_AB"], partitions["part_A"], partitions["part_B"]},
                encoding=encoding,
            ),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Check unique expand splits for subtree A
        unique_A = state.get_unique_expand_splits(partitions["part_A"])
        assert len(unique_A) == 1, (
            f"Subtree A should have 1 unique expand split, got {len(unique_A)}"
        )
        assert partitions["part_A"] in unique_A, "Subtree A should have [A] as unique"
        assert partitions["part_AB"] not in unique_A, (
            "[AB] should NOT be in unique (it's shared)"
        )

        # Check unique expand splits for subtree B
        unique_B = state.get_unique_expand_splits(partitions["part_B"])
        assert len(unique_B) == 1, (
            f"Subtree B should have 1 unique expand split, got {len(unique_B)}"
        )
        assert partitions["part_B"] in unique_B, "Subtree B should have [B] as unique"
        assert partitions["part_AB"] not in unique_B, (
            "[AB] should NOT be in unique (it's shared)"
        )

        print("✅ Unique expand splits correctly identified")

    def test_shared_expand_split_has_two_users(self, encoding, partitions):
        """
        Test that shared expand splits correctly track multiple users.

        Setup:
            - Both subtrees A and B need split [AB]

        Expected:
            - [AB] is in shared_expand_splits
            - [AB] has 2 users: {A, B}
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_AB"], partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_AB"], partitions["part_B"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_B"]}, encoding=encoding
            ),
        }

        state = InterpolationState(
            PartitionSet(
                {partitions["part_A"], partitions["part_B"]}, encoding=encoding
            ),
            PartitionSet(
                {partitions["part_AB"], partitions["part_A"], partitions["part_B"]},
                encoding=encoding,
            ),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Check that [AB] is shared
        assert partitions["part_AB"] in state.shared_expand_splits, (
            "[AB] should be in shared splits"
        )

        # Check that it has 2 users
        users = state.shared_expand_splits[partitions["part_AB"]]
        assert len(users) == 2, f"[AB] should have 2 users, got {len(users)}"
        assert partitions["part_A"] in users, "Subtree A should be a user of [AB]"
        assert partitions["part_B"] in users, "Subtree B should be a user of [AB]"

        print("✅ Shared expand split correctly has 2 users")

    def test_expand_last_strategy_initially_no_last_user(self, encoding, partitions):
        """
        Test the expand-last strategy: initially, no subtree is the last user.

        Setup:
            - Both subtrees need [AB]

        Expected:
            - get_expand_splits_for_last_user(A) returns [] (not last user yet)
            - get_expand_splits_for_last_user(B) returns [] (not last user yet)
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_AB"], partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_AB"], partitions["part_B"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_B"]}, encoding=encoding
            ),
        }

        state = InterpolationState(
            PartitionSet(
                {partitions["part_A"], partitions["part_B"]}, encoding=encoding
            ),
            PartitionSet(
                {partitions["part_AB"], partitions["part_A"], partitions["part_B"]},
                encoding=encoding,
            ),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Initially, neither is last user
        last_user_A = state.get_expand_splits_for_last_user(partitions["part_A"])
        last_user_B = state.get_expand_splits_for_last_user(partitions["part_B"])

        assert len(last_user_A) == 0, (
            f"Subtree A should NOT be last user initially, got {len(last_user_A)} splits"
        )
        assert len(last_user_B) == 0, (
            f"Subtree B should NOT be last user initially, got {len(last_user_B)} splits"
        )

        print("✅ Expand-last: Initially no last user")

    def test_expand_last_strategy_after_one_processed(self, encoding, partitions):
        """
        Test the expand-last strategy: after one subtree is processed, the other becomes last user.

        Setup:
            - Both subtrees need [AB]
            - Process subtree A (remove it from shared_expand_splits users)

        Expected:
            - get_expand_splits_for_last_user(B) returns [AB] (now last user)
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_AB"], partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_AB"], partitions["part_B"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_B"]}, encoding=encoding
            ),
        }

        state = InterpolationState(
            PartitionSet(
                {partitions["part_A"], partitions["part_B"]}, encoding=encoding
            ),
            PartitionSet(
                {partitions["part_AB"], partitions["part_A"], partitions["part_B"]},
                encoding=encoding,
            ),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Simulate processing subtree A: remove it from [AB]'s users
        state.shared_expand_splits[partitions["part_AB"]].discard(partitions["part_A"])

        # Now B should be the last user
        last_user_B = state.get_expand_splits_for_last_user(partitions["part_B"])

        assert len(last_user_B) == 1, (
            f"Subtree B should be last user now, got {len(last_user_B)} splits"
        )
        assert partitions["part_AB"] in last_user_B, (
            "Subtree B should have [AB] as last user split"
        )

        print("✅ Expand-last: After one processed, other becomes last user")

    def test_three_subtrees_shared_split(self, encoding, partitions):
        """
        Test with three subtrees sharing a split.

        Setup:
            - Subtrees A, B, C all need [ABC]

        Expected:
            - Initially, none are last user (count=3)
            - After A and B processed, C is last user
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_ABC"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_ABC"]}, encoding=encoding
            ),
            partitions["part_C"]: PartitionSet(
                {partitions["part_ABC"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {
            partitions["part_A"]: PartitionSet(set(), encoding=encoding),
            partitions["part_B"]: PartitionSet(set(), encoding=encoding),
            partitions["part_C"]: PartitionSet(set(), encoding=encoding),
        }

        state = InterpolationState(
            PartitionSet(set(), encoding=encoding),
            PartitionSet({partitions["part_ABC"]}, encoding=encoding),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Initially, none are last user
        assert len(state.get_expand_splits_for_last_user(partitions["part_A"])) == 0
        assert len(state.get_expand_splits_for_last_user(partitions["part_B"])) == 0
        assert len(state.get_expand_splits_for_last_user(partitions["part_C"])) == 0

        # Process A
        state.shared_expand_splits[partitions["part_ABC"]].discard(partitions["part_A"])
        assert len(state.get_expand_splits_for_last_user(partitions["part_B"])) == 0
        assert len(state.get_expand_splits_for_last_user(partitions["part_C"])) == 0

        # Process B
        state.shared_expand_splits[partitions["part_ABC"]].discard(partitions["part_B"])
        assert len(state.get_expand_splits_for_last_user(partitions["part_C"])) == 1

        print("✅ Three subtrees: Last user correctly identified")

    def test_categorize_splits_function(self, encoding, partitions):
        """
        Test the categorize_splits helper function directly.

        This is the core function that determines which splits are unique vs shared.
        """
        splits_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_AB"], partitions["part_A"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_AB"], partitions["part_B"]}, encoding=encoding
            ),
        }

        unique_splits, shared_splits = categorize_splits(splits_by_subtree)

        # Check unique splits
        assert len(unique_splits) == 2, (
            f"Should have 2 unique splits, got {len(unique_splits)}"
        )
        assert partitions["part_A"] in unique_splits, "[A] should be unique"
        assert partitions["part_B"] in unique_splits, "[B] should be unique"
        assert unique_splits[partitions["part_A"]] == partitions["part_A"], (
            "[A] owned by subtree A"
        )
        assert unique_splits[partitions["part_B"]] == partitions["part_B"], (
            "[B] owned by subtree B"
        )

        # Check shared splits
        assert len(shared_splits) == 1, (
            f"Should have 1 shared split, got {len(shared_splits)}"
        )
        assert partitions["part_AB"] in shared_splits, "[AB] should be shared"
        assert len(shared_splits[partitions["part_AB"]]) == 2, (
            "[AB] should have 2 users"
        )

        print("✅ categorize_splits function works correctly")

    def test_mixed_unique_and_shared_splits(self, encoding, partitions):
        """
        Test a complex scenario with both unique and shared splits.

        Setup:
            - Subtree A: unique [A], shared [AB]
            - Subtree B: unique [B], shared [AB], shared [BC]
            - Subtree C: unique [C], shared [BC]

        Expected:
            - Unique: {A: subtree_A, B: subtree_B, C: subtree_C}
            - Shared: {AB: {subtree_A, subtree_B}, BC: {subtree_B, subtree_C}}
        """
        expand_by_subtree = {
            partitions["part_A"]: PartitionSet(
                {partitions["part_A"], partitions["part_AB"]}, encoding=encoding
            ),
            partitions["part_B"]: PartitionSet(
                {partitions["part_B"], partitions["part_AB"], partitions["part_BC"]},
                encoding=encoding,
            ),
            partitions["part_C"]: PartitionSet(
                {partitions["part_C"], partitions["part_BC"]}, encoding=encoding
            ),
        }

        collapse_by_subtree = {}

        state = InterpolationState(
            PartitionSet(set(), encoding=encoding),
            PartitionSet(
                {
                    partitions["part_A"],
                    partitions["part_B"],
                    partitions["part_C"],
                    partitions["part_AB"],
                    partitions["part_BC"],
                },
                encoding=encoding,
            ),
            collapse_by_subtree,
            expand_by_subtree,
            partitions["part_ABCD"],
        )

        # Check unique splits
        assert partitions["part_A"] in state.unique_expand_splits
        assert partitions["part_B"] in state.unique_expand_splits
        assert partitions["part_C"] in state.unique_expand_splits

        # Check shared splits
        assert partitions["part_AB"] in state.shared_expand_splits
        assert partitions["part_BC"] in state.shared_expand_splits

        # Check users
        assert len(state.shared_expand_splits[partitions["part_AB"]]) == 2
        assert len(state.shared_expand_splits[partitions["part_BC"]]) == 2

        print("✅ Mixed unique and shared splits correctly categorized")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
